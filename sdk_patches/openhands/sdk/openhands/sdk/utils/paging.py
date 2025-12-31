"""Pagination utilities for iterating over paginated search results."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Protocol


class PageProtocol[T](Protocol):
    """Protocol for page objects returned by search functions.

    All page objects should have:
    - items: A list of items of type T
    - next_page_id: Optional string for pagination
    """

    items: list[T]
    next_page_id: str | None


async def page_iterator[T](
    search_func: Callable[..., Awaitable[PageProtocol[T]]],
    *args: Any,
    **kwargs: Any,
) -> AsyncGenerator[T, None]:
    """
    Iterate over items from paginated search results.

    This utility function handles pagination automatically by calling the search
    function repeatedly with updated page_id parameters until all pages are
    exhausted.

    Args:
        search_func: An async function that returns a PageProtocol[T] object
                    with 'items' and 'next_page_id' attributes
        *args: Positional arguments to pass to the search function
        **kwargs: Keyword arguments to pass to the search function

    Yields:
        Individual items of type T from each page

    Example:
        async for event in page_iterator(event_service.search_events, limit=50):
            await send_event(event, websocket)

        async for conversation in page_iterator(
            conversation_service.search_conversations,
            execution_status=ConversationExecutionStatus.RUNNING
        ):
            print(conversation.title)
    """
    page_id = kwargs.pop("page_id", None)

    while True:
        # Call the search function with current page_id
        page = await search_func(*args, page_id=page_id, **kwargs)

        # Yield each item from the current page
        for item in page.items:
            yield item

        # Check if there are more pages
        page_id = page.next_page_id
        if not page_id:
            break
