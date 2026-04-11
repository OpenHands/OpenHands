"""Pagination utilities for OpenHands App Server.

This module provides utilities for handling pagination in API responses.
The page_id is designed to be an opaque token - external consumers should not
infer details of the underlying pagination strategy (e.g., offset vs page number)
from the page_id contents, as this may change in the future.
"""

import base64


def encode_page_id(value: int) -> str:
    """Encode an integer page identifier as an opaque base64 string.

    This function exists to ensure that page IDs are opaque tokens that do not
    reveal the underlying pagination strategy to API consumers. The encoding
    format may change in the future, so consumers should treat page_id as an
    opaque string and not attempt to parse or construct it themselves.

    Args:
        value: The integer value to encode.

    Returns:
        Base64-encoded string (URL-safe, without padding).
    """
    return base64.urlsafe_b64encode(str(value).encode()).decode().rstrip('=')


def decode_page_id(page_id: str | None) -> int | None:
    """Decode an opaque page identifier back to an integer.

    This function exists to support the internal decoding of page IDs.
    External consumers should not need to call this function directly - they
    should pass the page_id string received from a previous response back to
    the API to get the next page of results.

    Note: The encoding format may change in the future, so this function
    should not be used to construct page_ids externally.

    Args:
        page_id: The base64-encoded page ID string.

    Returns:
        The decoded integer value, or None if page_id is None/empty.
    """
    if not page_id:
        return None
    try:
        # Add padding back if needed
        padded = page_id + '=' * (4 - len(page_id) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode()).decode()
        return int(decoded)
    except (ValueError, Exception):
        return None


def paginate_results(
    items: list, page_id: str | None, limit: int
) -> tuple[list, str | None]:
    """Apply pagination to a list of items.

    Args:
        items: Full list of items to paginate.
        page_id: Optional page token from previous request.
        limit: Maximum number of items per page.

    Returns:
        Tuple of (paginated_items, next_page_id).
    """
    start_offset = 0
    decoded_page_id = decode_page_id(page_id)
    if decoded_page_id is not None:
        start_offset = decoded_page_id

    end_offset = start_offset + limit
    paginated_items = items[start_offset:end_offset]
    next_page_id = None
    if end_offset < len(items):
        next_page_id = encode_page_id(end_offset)

    return paginated_items, next_page_id
