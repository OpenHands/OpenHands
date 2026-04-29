from dataclasses import dataclass

import pytest

from openhands.utils.search_utils import iterate, offset_to_page_id, page_id_to_offset


@dataclass
class MockItem:
    id: str


@dataclass
class MockResultSet:
    results: list[MockItem]
    next_page_id: str | None = None


def test_offset_to_page_id():
    # Test with has_next=True
    assert bool(offset_to_page_id(10, True))
    assert bool(offset_to_page_id(0, True))

    # Test with has_next=False should return None
    assert offset_to_page_id(10, False) is None
    assert offset_to_page_id(0, False) is None


def test_page_id_to_offset():
    # Test with None should return 0
    assert page_id_to_offset(None) == 0


def test_bidirectional_conversion():
    # Test converting offset to page_id and back
    test_offsets = [0, 1, 10, 100, 1000]
    for offset in test_offsets:
        page_id = offset_to_page_id(offset, True)
        assert page_id_to_offset(page_id) == offset


@pytest.mark.asyncio
async def test_iterate_empty():
    async def mock_search(page_id=None, limit=20):
        return MockResultSet(results=[])

    results = []
    async for result in iterate(mock_search):
        results.append(result)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_iterate_single_page():
    items = [MockItem(id='item1'), MockItem(id='item2')]

    async def mock_search(page_id=None, limit=20):
        return MockResultSet(results=items, next_page_id=None)

    results = []
    async for result in iterate(mock_search):
        results.append(result)

    assert len(results) == 2
    assert results[0].id == 'item1'
    assert results[1].id == 'item2'


@pytest.mark.asyncio
async def test_iterate_multiple_pages():
    # Create test data with 5 items split across pages
    all_items = [MockItem(id=f'item{i}') for i in range(1, 6)]

    async def mock_search(page_id=None, limit=2):
        offset = page_id_to_offset(page_id)
        end = min(offset + limit, len(all_items))
        items = all_items[offset:end]
        has_next = end < len(all_items)
        next_page = offset_to_page_id(end, has_next)
        return MockResultSet(results=items, next_page_id=next_page)

    results = []
    async for result in iterate(mock_search, limit=2):
        results.append(result)

    assert len(results) == 5
    assert [r.id for r in results] == ['item1', 'item2', 'item3', 'item4', 'item5']
