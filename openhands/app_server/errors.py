from typing import Any

from fastapi import HTTPException, status


class OpenHandsError(HTTPException):
    """General Error"""

    def __init__(
        self,
        detail: Any = None,
        headers: dict[str, str] | None = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class AuthError(OpenHandsError):
    """Error in authentication."""

    def __init__(
        self,
        detail: Any = None,
        headers: dict[str, str] | None = None,
        status_code: int = status.HTTP_401_UNAUTHORIZED,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class PermissionsError(OpenHandsError):
    """Error in permissions."""

    def __init__(
        self,
        detail: Any = None,
        headers: dict[str, str] | None = None,
        status_code: int = status.HTTP_403_FORBIDDEN,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class SandboxError(OpenHandsError):
    """Error in Sandbox."""


class SandboxDeleteRetryError(OpenHandsError):
    """The sandbox exists but its delete could not complete and was kept for retry.

    Raised by ``delete_sandbox`` when the runtime /stop or lookup hits a transient
    failure. (Archiving never raises — it returns False from
    ``archive_conversation_workspace`` to signal a REQUIRED capture should block.)
    503 (vs 404) so a client distinguishes "still here, try again" from "not
    found" and keeps retrying.
    """

    def __init__(
        self,
        detail: Any = None,
        headers: dict[str, str] | None = None,
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
