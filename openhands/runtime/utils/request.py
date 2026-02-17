# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import json
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from openhands.utils.http_session import HttpSession
from openhands.utils.tenacity_stop import stop_if_should_exit


class RequestHTTPError(httpx.HTTPStatusError):
    """Exception raised when an error occurs in a request with details."""

    def __init__(self, *args: Any, detail: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.detail = detail

    def __str__(self) -> str:
        s = super().__str__()
        if self.detail is not None:
            s += f'\nDetails: {self.detail}'
        return str(s)


def is_retryable_error(exception: Any) -> bool:
    return (
        isinstance(exception, httpx.HTTPStatusError)
        and exception.response.status_code == 429
    )


@retry(
    retry=retry_if_exception(is_retryable_error),
    stop=stop_after_attempt(3) | stop_if_should_exit(),
    wait=wait_exponential(multiplier=1, min=4, max=60),
)
def send_request(
    session: HttpSession,
    method: str,
    url: str,
    timeout: int = 60,
    **kwargs: Any,
) -> httpx.Response:
    response = session.request(method, url, timeout=timeout, **kwargs)
    try:
        response.raise_for_status()
    except httpx.HTTPError as e:
        try:
            _json = response.json()
        except json.decoder.JSONDecodeError:
            _json = None
        finally:
            response.close()
        raise RequestHTTPError(
            e,
            request=e.request,
            response=e.response,
            detail=_json.get('detail') if _json is not None else None,
        ) from e
    return response
