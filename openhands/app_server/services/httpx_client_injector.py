from typing import AsyncGenerator

import httpx
from fastapi import Request
from pydantic import BaseModel, Field

from openhands.app_server.services.injector import Injector, InjectorState

HTTPX_CLIENT_ATTR = 'httpx_client'
HTTPX_CLIENT_KEEP_OPEN_ATTR = 'httpx_client_keep_open'


class HttpxClientInjector(BaseModel, Injector[httpx.AsyncClient]):
    """Injector for a httpx client. By keeping a single httpx client alive in the
    context of server requests handshakes are minimized while connection pool leaks
    are prevented."""

    timeout: int = Field(default=15, description='Default timeout on all http requests')

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[httpx.AsyncClient, None]:
        httpx_client = getattr(state, HTTPX_CLIENT_ATTR, None)
        if httpx_client:
            yield httpx_client
            return
        httpx_client = httpx.AsyncClient(timeout=self.timeout)
        try:
            setattr(state, HTTPX_CLIENT_ATTR, httpx_client)
            yield httpx_client
        finally:
            # If instructed, do not close the httpx client at the end of the request.
            if not getattr(state, HTTPX_CLIENT_KEEP_OPEN_ATTR, False):
                # Clean up the httpx client from request state
                if hasattr(state, HTTPX_CLIENT_ATTR):
                    delattr(state, HTTPX_CLIENT_ATTR)
                await httpx_client.aclose()


def set_httpx_client_keep_open(state: InjectorState, keep_open: bool):
    """Set whether the connection should be kept open after the request terminates."""
    setattr(state, HTTPX_CLIENT_KEEP_OPEN_ATTR, keep_open)
