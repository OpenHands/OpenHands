import base64
from typing import AsyncIterator, Callable


def offset_to_page_id(offset: int, has_next: bool) -> str | None:
    if not has_next:
        return None
    next_page_id = base64.b64encode(str(offset).encode()).decode()
    return next_page_id


def page_id_to_offset(page_id: str | None) -> int:
    if not page_id:
        return 0
    offset = int(base64.b64decode(page_id).decode())
    return offset


async def iterate(fn: Callable, **kwargs) -> AsyncIterator:
    """Iterate over paged result sets. Assumes that the results sets contain an array of result objects, and a next_page_id"""
    kwargs = {**kwargs}
    kwargs['page_id'] = None
    while True:
        result_set = await fn(**kwargs)
        # Handle both 'results' and 'items' attributes for different model types
        items = getattr(result_set, 'results', None) or getattr(result_set, 'items', [])
        for result in items:
            yield result
        if result_set.next_page_id is None:
            return
        kwargs['page_id'] = result_set.next_page_id
