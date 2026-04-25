from __future__ import annotations

from typing import Any, Iterator
from uuid import UUID

from integrations.utils import CONVERSATION_URL

_ACK_MARKER_TEMPLATE = '<!-- openhands-ack:{} -->'


def build_ack_marker(conversation_id: UUID | str) -> str:
    """Build a reusable GitHub acknowledgement marker for a resolver conversation."""
    return _ACK_MARKER_TEMPLATE.format(conversation_id)


def append_ack_marker(message: str, conversation_id: UUID | str) -> str:
    """Append the acknowledgement marker to a message body if it is not already present."""
    marker = build_ack_marker(conversation_id)
    if marker in message:
        return message

    if message.endswith('\n'):
        return f'{message}{marker}'

    return f'{message}\n\n{marker}'


def ensure_conversation_link(message: str, conversation_id: UUID | str) -> str:
    """Ensure the conversation link is included in the message body."""
    link_fragment = f'conversations/{conversation_id}'
    if link_fragment in message:
        return message

    conversation_link = CONVERSATION_URL.format(conversation_id)
    return f'{message.rstrip()}\n\nTrack progress [here]({conversation_link})'


def build_final_resolver_comment(summary: str, conversation_id: UUID | str) -> str:
    """Build the final resolver comment with link and acknowledgement marker."""
    summary_with_link = ensure_conversation_link(summary.strip(), conversation_id)
    return append_ack_marker(summary_with_link, conversation_id)


def iter_recent_paginated_items(
    paginated_list: Any, max_items: int | None = None
) -> Iterator[Any]:
    """Iterate over a GitHub paginated list from newest to oldest.

    This assumes the underlying pages are ordered oldest-first, and it iterates
    pages in reverse, yielding each page's items in reverse order.
    """
    page_count = getattr(paginated_list, 'totalCount', None)
    if page_count is None:
        raise ValueError('Paginated list does not expose totalCount')

    yielded = 0
    for page_index in range(page_count - 1, -1, -1):
        page = paginated_list.get_page(page_index)
        for item in reversed(page):
            yield item
            yielded += 1
            if max_items is not None and yielded >= max_items:
                return
