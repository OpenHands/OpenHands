class AuthError(Exception):
    """Generic auth error."""


class NoCredentialsError(AuthError):
    """Error when no authentication was provided."""


class EmailNotVerifiedError(AuthError):
    """Error when email is not verified."""


class BearerTokenError(AuthError):
    """Error when decoding a bearer token."""


class CookieError(AuthError):
    """Error when decoding an auth cookie."""


class TosNotAcceptedError(AuthError):
    """Error when decoding an auth cookie."""


class ExpiredError(AuthError):
    """Error when a token has expired (Usually the refresh token)."""


class TokenRefreshError(AuthError):
    """Error when token refresh fails due to timeout or lock contention."""
