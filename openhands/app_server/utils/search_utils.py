import base64
import binascii
from typing import AsyncIterator, Callable


def offset_to_page_id(offset: int, has_next: bool) -> str | None:
    if not has_next:
        return None
    next_page_id = base64.b64encode(str(offset).encode()).decode()
    return next_page_id


def page_id_to_offset(page_id: str | None) -> int:
    if not page_id:
        return 0
    try:
        return int(base64.b64decode(page_id).decode())
    except (ValueError, UnicodeDecodeError, binascii.Error):
        # Malformed/opaque page_id -> start from the beginning, mirroring
        # paging_utils.decode_page_id which also tolerates bad cursors.
        return 0


async def iterate(fn: Callable, **kwargs) -> AsyncIterator:
    """Iterate over paged result sets. Assumes that the results sets contain an array of result objects, and a next_page_id"""
    kwargs = {**kwargs}
    kwargs['page_id'] = None
    while True:
        result_set = await fn(**kwargs)
        items = getattr(result_set, 'items', None)
        if items is None:
            items = getattr(result_set, 'results')
        for result in items:
            yield result
        if result_set.next_page_id is None:
            return
        kwargs['page_id'] = result_set.next_page_id
