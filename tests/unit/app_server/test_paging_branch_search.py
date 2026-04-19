"""Tests for branch search pagination tokens (V1 git API)."""

from openhands.app_server.utils.paging_utils import (
    decode_branch_search_continuation,
    encode_github_branch_search_continuation,
    encode_page_id,
)


def test_decode_branch_search_none_is_first_page():
    page, after = decode_branch_search_continuation(None)
    assert page == 1
    assert after is None


def test_decode_branch_search_integer_page():
    tok = encode_page_id(3)
    page, after = decode_branch_search_continuation(tok)
    assert page == 3
    assert after is None


def test_github_cursor_token_roundtrip():
    tok = encode_github_branch_search_continuation('cursor_abc')
    page, after = decode_branch_search_continuation(tok)
    assert page == 1
    assert after == 'cursor_abc'
