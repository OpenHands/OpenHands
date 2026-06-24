from unittest.mock import MagicMock, patch

import httpx
from integrations.slack.slack_attachments import (
    MAX_SLACK_ATTACHMENT_BYTES,
    MAX_SLACK_ATTACHMENTS_PER_MESSAGE,
    collect_message_attachment_content,
)
from slack_sdk.errors import SlackApiError


def test_slack_authorize_urls_request_files_read_scope():
    from integrations.slack.slack_manager import authorize_url_generator
    from server.routes.integration.slack import (
        authorize_url_generator as install_authorize_url_generator,
    )

    assert 'files:read' in authorize_url_generator.scopes
    assert 'files:read' in install_authorize_url_generator.scopes


def _mock_download(mock_client_cls: MagicMock, content: bytes) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.raise_for_status = MagicMock()
    http_client = mock_client_cls.return_value.__enter__.return_value
    http_client.get.return_value = response
    return http_client


@patch('integrations.slack.slack_attachments.httpx.Client')
def test_collect_message_attachment_content_includes_image_data_url(mock_client_cls):
    http_client = _mock_download(mock_client_cls, b'image-bytes')
    slack_client = MagicMock()

    content = collect_message_attachment_content(
        slack_client,
        'xoxb-test-token',
        {
            'files': [
                {
                    'title': 'screenshot.png',
                    'mimetype': 'image/png',
                    'size': 11,
                    'url_private': 'https://files.slack.com/files-pri/screenshot.png',
                }
            ]
        },
    )

    assert content.image_urls == ['data:image/png;base64,aW1hZ2UtYnl0ZXM=']
    assert 'screenshot.png' in content.descriptions[0]
    assert 'included as image content' in content.descriptions[0]
    http_client.get.assert_called_once_with(
        'https://files.slack.com/files-pri/screenshot.png',
        headers={'Authorization': 'Bearer xoxb-test-token'},
    )


@patch('integrations.slack.slack_attachments.httpx.Client')
def test_collect_message_attachment_content_extracts_text_files(mock_client_cls):
    _mock_download(mock_client_cls, b'apiVersion: v1\nkind: ConfigMap\n')
    slack_client = MagicMock()

    content = collect_message_attachment_content(
        slack_client,
        'xoxb-test-token',
        {
            'files': [
                {
                    'title': 'config.yaml',
                    'mimetype': 'application/x-yaml',
                    'url_private': 'https://files.slack.com/files-pri/config.yaml',
                }
            ]
        },
    )

    assert content.image_urls == []
    assert 'config.yaml' in content.descriptions[0]
    assert 'apiVersion: v1' in content.descriptions[0]
    assert 'kind: ConfigMap' in content.descriptions[0]


@patch('integrations.slack.slack_attachments.httpx.Client')
def test_collect_message_attachment_content_hydrates_file_info(mock_client_cls):
    _mock_download(mock_client_cls, b'log line')
    slack_client = MagicMock()
    slack_client.files_info.return_value = {
        'file': {
            'title': 'debug.log',
            'mimetype': 'text/plain',
            'url_private': 'https://files.slack.com/files-pri/debug.log',
        }
    }

    content = collect_message_attachment_content(
        slack_client,
        'xoxb-test-token',
        {'files': [{'id': 'F123'}]},
    )

    slack_client.files_info.assert_called_once_with(file='F123')
    assert 'debug.log' in content.descriptions[0]
    assert 'log line' in content.descriptions[0]


@patch('integrations.slack.slack_attachments.httpx.Client')
def test_collect_message_attachment_content_isolates_download_errors(mock_client_cls):
    http_client = mock_client_cls.return_value.__enter__.return_value
    http_client.get.side_effect = httpx.TimeoutException('timeout')
    slack_client = MagicMock()

    content = collect_message_attachment_content(
        slack_client,
        'xoxb-test-token',
        {
            'files': [
                {
                    'title': 'flaky.png',
                    'mimetype': 'image/png',
                    'url_private': 'https://files.slack.com/files-pri/flaky.png',
                }
            ]
        },
    )

    assert content.image_urls == []
    assert content.descriptions == ['- flaky.png could not be read due to an error.']


def test_collect_message_attachment_content_isolates_non_scope_slack_api_errors():
    slack_client = MagicMock()
    slack_client.files_info.side_effect = SlackApiError(
        message='Slack unavailable', response={'error': 'internal_error'}
    )

    content = collect_message_attachment_content(
        slack_client,
        'xoxb-test-token',
        {'files': [{'id': 'F123', 'title': 'metadata-only.txt'}]},
    )

    assert content.image_urls == []
    assert content.descriptions == [
        '- metadata-only.txt could not be read due to an error.'
    ]


@patch('integrations.slack.slack_attachments.httpx.Client')
def test_collect_message_attachment_content_caps_attachments_per_message(
    mock_client_cls,
):
    _mock_download(mock_client_cls, b'content')
    slack_client = MagicMock()

    content = collect_message_attachment_content(
        slack_client,
        'xoxb-test-token',
        {
            'files': [
                {
                    'title': f'file-{index}.txt',
                    'mimetype': 'text/plain',
                    'url_private': f'https://files.slack.com/files-pri/file-{index}.txt',
                }
                for index in range(MAX_SLACK_ATTACHMENTS_PER_MESSAGE + 2)
            ]
        },
    )

    assert len(content.descriptions) == MAX_SLACK_ATTACHMENTS_PER_MESSAGE + 1
    assert content.descriptions[-1] == (
        '- 2 additional Slack attachment(s) were skipped because only '
        f'{MAX_SLACK_ATTACHMENTS_PER_MESSAGE} attachments per message are read.'
    )
    http_client = mock_client_cls.return_value.__enter__.return_value
    assert http_client.get.call_count == MAX_SLACK_ATTACHMENTS_PER_MESSAGE


def test_collect_message_attachment_content_rejects_oversized_files_before_download():
    slack_client = MagicMock()

    content = collect_message_attachment_content(
        slack_client,
        'xoxb-test-token',
        {
            'files': [
                {
                    'title': 'huge.log',
                    'mimetype': 'text/plain',
                    'size': MAX_SLACK_ATTACHMENT_BYTES + 1,
                    'url_private': 'https://files.slack.com/files-pri/huge.log',
                }
            ]
        },
    )

    assert content.image_urls == []
    assert 'huge.log' in content.descriptions[0]
    assert 'exceeds the Slack attachment size limit' in content.descriptions[0]
