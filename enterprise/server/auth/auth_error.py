class AuthError(Exception):
    """Generic auth error"""

    pass


class NoCredentialsError(AuthError):
    """Error when no authentication was provided"""

    pass


class EmailNotVerifiedError(AuthError):
    """Error when email is not verified"""

    pass


class BearerTokenError(AuthError):
    """Error when decoding a bearer token"""

    pass


class CookieError(AuthError):
    """Error when decoding an auth cookie"""

    pass


class TosNotAcceptedError(AuthError):
    """Error when decoding an auth cookie"""

    pass


class ExpiredError(AuthError):
    """Error when a token has expired (Usually the refresh token)"""

    pass


class TokenRefreshError(AuthError):
    """Error when token refresh fails due to timeout or lock contention"""

    pass
