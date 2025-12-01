"""Utility for injecting credentials into HTTP requests automatically."""

from __future__ import annotations

import httpx
from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger

if TYPE_CHECKING:
    from openhands.storage.credentials.resolver import CredentialResolver


async def send_request_with_credential_retry(
    resolver: CredentialResolver | None,
    session: httpx.AsyncClient | httpx.Client,
    method: str,
    url: str,
    **kwargs: any,
) -> httpx.Response:
    """Send an HTTP request with automatic credential injection on 401/403 errors.

    This function attempts to send a request, and if it receives a 401 or 403
    response, it automatically resolves credentials for the URL, injects them
    as headers, and retries the request.

    Args:
        resolver: CredentialResolver instance (optional, will skip retry if None)
        session: httpx client (AsyncClient or Client)
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        **kwargs: Additional arguments to pass to session.request()

    Returns:
        httpx.Response object

    Raises:
        httpx.HTTPStatusError: If the request fails after retry with credentials
    """
    # Make the initial request
    if isinstance(session, httpx.AsyncClient):
        response = await session.request(method, url, **kwargs)
    else:
        response = session.request(method, url, **kwargs)

    # Check for authentication errors
    if response.status_code in (401, 403):
        if resolver:
            logger.debug(
                f'Received {response.status_code} for {url}, attempting credential injection'
            )
            result = resolver.resolve_credential(url)

            if result:
                credential_value, auth_headers = result
                logger.info(
                    f'Resolved credentials for {url}, retrying request with auth headers'
                )

                # Update headers with credential headers
                existing_headers = kwargs.get('headers', {}) or {}
                existing_headers.update(auth_headers)
                kwargs['headers'] = existing_headers

                # Close the previous response before retrying
                response.close()

                # Retry the request with credentials
                if isinstance(session, httpx.AsyncClient):
                    response = await session.request(method, url, **kwargs)
                else:
                    response = session.request(method, url, **kwargs)

                if response.status_code in (401, 403):
                    logger.warning(
                        f'Request to {url} still failed with {response.status_code} after credential injection'
                    )
                else:
                    logger.info(
                        f'Successfully authenticated request to {url} using stored credentials'
                    )
            else:
                logger.debug(
                    f'No credentials found for {url}, returning original {response.status_code} response'
                )
        else:
            logger.debug(
                f'No credential resolver available for {url}, returning original {response.status_code} response'
            )

    return response

