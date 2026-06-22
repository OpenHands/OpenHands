"""Helpers for reading Slack file attachments for agent context."""

import base64
import mimetypes
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import httpx
from integrations.slack.slack_errors import SlackError, SlackErrorCode
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from openhands.app_server.utils.logger import openhands_logger as logger

MAX_SLACK_ATTACHMENTS_PER_MESSAGE = 5
MAX_SLACK_ATTACHMENT_BYTES = 5 * 1024 * 1024
MAX_ATTACHMENT_TEXT_CHARS = 50_000

TEXT_FILE_EXTENSIONS = {
    '.bash',
    '.cfg',
    '.conf',
    '.config',
    '.css',
    '.csv',
    '.env',
    '.go',
    '.html',
    '.ini',
    '.java',
    '.js',
    '.json',
    '.jsx',
    '.log',
    '.md',
    '.properties',
    '.py',
    '.rb',
    '.rs',
    '.sh',
    '.sql',
    '.toml',
    '.ts',
    '.tsx',
    '.txt',
    '.xml',
    '.yaml',
    '.yml',
}

PDF_MIME_TYPES = {'application/pdf'}
DOCX_MIME_TYPES = {
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}


@dataclass
class SlackAttachmentContent:
    descriptions: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)


def collect_message_attachment_content(
    slack_client: WebClient,
    bot_access_token: str,
    message: dict[str, Any],
) -> SlackAttachmentContent:
    """Download and convert Slack file attachments attached to a message."""
    files = message.get('files') or []
    content = SlackAttachmentContent()
    if not isinstance(files, list) or not files:
        return content

    for file_info in files[:MAX_SLACK_ATTACHMENTS_PER_MESSAGE]:
        if not isinstance(file_info, dict):
            continue
        try:
            attachment = _collect_file_attachment_content(
                slack_client, bot_access_token, file_info
            )
        except SlackError:
            raise
        except Exception as e:
            logger.warning(
                'slack_attachment_collect_failed',
                extra={
                    'file_id': file_info.get('id'),
                    'file_title': _attachment_title(file_info),
                    'error': str(e),
                },
            )
            attachment = SlackAttachmentContent(
                descriptions=[
                    f'- {_attachment_title(file_info)} could not be read due to an error.'
                ]
            )
        content.descriptions.extend(attachment.descriptions)
        content.image_urls.extend(attachment.image_urls)

    skipped_count = len(files) - MAX_SLACK_ATTACHMENTS_PER_MESSAGE
    if skipped_count > 0:
        content.descriptions.append(
            f'- {skipped_count} additional Slack attachment(s) were skipped because '
            f'only {MAX_SLACK_ATTACHMENTS_PER_MESSAGE} attachments per message are read.'
        )

    return content


def _collect_file_attachment_content(
    slack_client: WebClient,
    bot_access_token: str,
    file_info: dict[str, Any],
) -> SlackAttachmentContent:
    file_info = _hydrate_file_info(slack_client, file_info)
    title = _attachment_title(file_info)
    mimetype = _attachment_mimetype(file_info, title)
    size = _safe_int(file_info.get('size'))
    size_label = _format_size(size)

    if size and size > MAX_SLACK_ATTACHMENT_BYTES:
        return SlackAttachmentContent(
            descriptions=[
                f'- {title} ({mimetype or "unknown type"}, {size_label}) was not '
                'read because it exceeds the Slack attachment size limit.'
            ]
        )

    download_url = file_info.get('url_private_download') or file_info.get('url_private')
    if not download_url:
        return SlackAttachmentContent(
            descriptions=[
                f'- {title} ({mimetype or "unknown type"}, {size_label}) could not '
                'be downloaded from Slack.'
            ]
        )

    file_bytes = _download_slack_file(bot_access_token, download_url)
    if len(file_bytes) > MAX_SLACK_ATTACHMENT_BYTES:
        return SlackAttachmentContent(
            descriptions=[
                f'- {title} ({mimetype or "unknown type"}, '
                f'{_format_size(len(file_bytes))}) was not read because it exceeds '
                'the Slack attachment size limit.'
            ]
        )

    if mimetype.startswith('image/'):
        data_url = _to_data_url(mimetype, file_bytes)
        return SlackAttachmentContent(
            descriptions=[
                f'- {title} ({mimetype}, {_format_size(len(file_bytes))}) is included '
                'as image content.'
            ],
            image_urls=[data_url],
        )

    extracted_text = _extract_attachment_text(file_bytes, title, mimetype)
    if extracted_text:
        return SlackAttachmentContent(
            descriptions=[
                f'- {title} ({mimetype or "unknown type"}, '
                f'{_format_size(len(file_bytes))}) content:\n{_truncate_text(extracted_text)}'
            ]
        )

    return SlackAttachmentContent(
        descriptions=[
            f'- {title} ({mimetype or "unknown type"}, {_format_size(len(file_bytes))}) '
            'was downloaded from Slack, but this file type is not currently readable '
            'as text or image context.'
        ]
    )


def _hydrate_file_info(
    slack_client: WebClient, file_info: dict[str, Any]
) -> dict[str, Any]:
    if file_info.get('url_private') or file_info.get('url_private_download'):
        return file_info

    file_id = file_info.get('id')
    if not file_id:
        return file_info

    try:
        response = slack_client.files_info(file=file_id)
    except SlackApiError as e:
        if e.response.get('error') == 'missing_scope':
            raise SlackError(SlackErrorCode.MISSING_SLACK_SCOPES) from e
        raise

    detailed_file = response.get('file') or {}
    return {**file_info, **detailed_file}


def _download_slack_file(bot_access_token: str, url: str) -> bytes:
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={'Authorization': f'Bearer {bot_access_token}'},
            )
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        if e.response.status_code in {401, 403}:
            raise SlackError(SlackErrorCode.MISSING_SLACK_SCOPES) from e
        raise


def _extract_attachment_text(
    file_bytes: bytes,
    title: str,
    mimetype: str,
) -> str | None:
    if _is_text_attachment(title, mimetype):
        return file_bytes.decode('utf-8', errors='replace').strip()
    if mimetype in PDF_MIME_TYPES:
        return _extract_pdf_text(file_bytes)
    if mimetype in DOCX_MIME_TYPES:
        return _extract_docx_text(file_bytes)
    return None


def _extract_pdf_text(file_bytes: bytes) -> str | None:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        text = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
        return text.strip() or None
    except Exception as e:
        logger.warning('slack_attachment_pdf_extract_failed', extra={'error': str(e)})
        return None


def _extract_docx_text(file_bytes: bytes) -> str | None:
    try:
        from docx import Document

        document = Document(BytesIO(file_bytes))
        text = '\n'.join(paragraph.text for paragraph in document.paragraphs)
        return text.strip() or None
    except Exception as e:
        logger.warning('slack_attachment_docx_extract_failed', extra={'error': str(e)})
        return None


def _is_text_attachment(title: str, mimetype: str) -> bool:
    if mimetype.startswith('text/'):
        return True
    extension = _attachment_extension(title)
    return extension in TEXT_FILE_EXTENSIONS


def _to_data_url(mimetype: str, file_bytes: bytes) -> str:
    encoded = base64.b64encode(file_bytes).decode('ascii')
    return f'data:{mimetype};base64,{encoded}'


def _truncate_text(text: str) -> str:
    if len(text) <= MAX_ATTACHMENT_TEXT_CHARS:
        return text
    return (
        text[:MAX_ATTACHMENT_TEXT_CHARS]
        + f'\n\n[Slack attachment text truncated after {MAX_ATTACHMENT_TEXT_CHARS} characters]'
    )


def _attachment_title(file_info: dict[str, Any]) -> str:
    return str(
        file_info.get('title')
        or file_info.get('name')
        or file_info.get('id')
        or 'Slack attachment'
    )


def _attachment_mimetype(file_info: dict[str, Any], title: str) -> str:
    mimetype = file_info.get('mimetype')
    if isinstance(mimetype, str) and mimetype:
        return mimetype
    guessed_mimetype, _ = mimetypes.guess_type(title)
    return guessed_mimetype or ''


def _attachment_extension(title: str) -> str:
    _, _, extension = title.rpartition('.')
    if not extension or extension == title:
        return ''
    return f'.{extension.lower()}'


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_size(size: int | None) -> str:
    if size is None:
        return 'unknown size'
    if size < 1024:
        return f'{size} bytes'
    if size < 1024 * 1024:
        return f'{size / 1024:.1f} KB'
    return f'{size / (1024 * 1024):.1f} MB'
