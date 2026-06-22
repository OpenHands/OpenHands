from unittest.mock import MagicMock, patch

from integrations.slack.slack_attachments import collect_message_attachment_content


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
