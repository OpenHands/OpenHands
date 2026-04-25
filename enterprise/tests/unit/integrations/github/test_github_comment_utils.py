from uuid import uuid4

from integrations.github.github_comment_utils import (
    append_ack_marker,
    build_ack_marker,
    build_final_resolver_comment,
    iter_recent_paginated_items,
)


class FakePaginatedList:
    def __init__(self, pages):
        self.pages = pages
        self.totalCount = len(pages)

    def get_page(self, index):
        return self.pages[index]


def test_append_ack_marker_is_idempotent():
    conversation_id = uuid4()
    message = "I'm on it!"
    first = append_ack_marker(message, conversation_id)
    second = append_ack_marker(first, conversation_id)

    assert first == second
    assert first.count(build_ack_marker(conversation_id)) == 1


def test_build_final_resolver_comment_preserves_conversation_link_and_marker():
    conversation_id = uuid4()
    summary = 'Resolved the issue.'

    final_comment = build_final_resolver_comment(summary, conversation_id)

    assert f'conversations/{conversation_id}' in final_comment
    assert build_ack_marker(conversation_id) in final_comment
    assert final_comment.startswith(summary)


def test_iter_recent_paginated_items_yields_newest_first():
    pages = [[1, 2], [3, 4], [5, 6]]
    paginated = FakePaginatedList(pages)

    items = list(iter_recent_paginated_items(paginated))

    assert items == [6, 5, 4, 3, 2, 1]


def test_iter_recent_paginated_items_traverses_older_pages_when_needed():
    pages = [['a'], ['b'], ['c']]
    paginated = FakePaginatedList(pages)

    items = list(iter_recent_paginated_items(paginated, max_items=2))

    assert items == ['c', 'b']
