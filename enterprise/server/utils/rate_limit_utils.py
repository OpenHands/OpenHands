from fastapi import HTTPException, Request, status

from openhands.core.logger import openhands_logger as logger
from openhands.server.shared import sio

# Rate limiting constants
RATE_LIMIT_USER_SECONDS = 120  # 2 minutes per user_id
RATE_LIMIT_IP_SECONDS = 300  # 5 minutes per IP address


async def check_rate_limit_by_user_id(
    request: Request,
    key_prefix: str,
    user_id: str | None,
    user_rate_limit_seconds: int = RATE_LIMIT_USER_SECONDS,
    ip_rate_limit_seconds: int = RATE_LIMIT_IP_SECONDS,
) -> None:
    """
    Check rate limit for requests, using user_id when available, falling back to IP address.

    Uses Redis to store rate limit keys with expiration. If a key already exists,
    it means the rate limit is active and the request will be rejected.

    Args:
        request: FastAPI Request object
        key_prefix: Prefix for the Redis key (e.g., "email_resend")
        user_id: User ID if available, None otherwise
        user_rate_limit_seconds: Rate limit window in seconds for user_id-based limiting (default: 120)
        ip_rate_limit_seconds: Rate limit window in seconds for IP-based limiting (default: 300)

    Raises:
        HTTPException: If rate limit is exceeded (429 status code)
    """
    try:
        redis = sio.manager.redis
        if not redis:
            # If Redis is unavailable, log warning and allow request (fail open)
            logger.warning('Redis unavailable for rate limiting, allowing request')
            return

        if user_id:
            # Rate limit by user_id (primary method)
            rate_limit_key = f'{key_prefix}:{user_id}'
            rate_limit_seconds = user_rate_limit_seconds
        else:
            # Fallback to IP address rate limiting
            client_ip = request.client.host if request.client else 'unknown'
            rate_limit_key = f'{key_prefix}:ip:{client_ip}'
            rate_limit_seconds = ip_rate_limit_seconds

        # Try to set the key with expiration. If it already exists (nx=True fails),
        # it means the rate limit is active
        created = await redis.set(rate_limit_key, 1, nx=True, ex=rate_limit_seconds)

        if not created:
            logger.info(
                f'Rate limit exceeded for {rate_limit_key}',
                extra={
                    'user_id': user_id,
                    'ip': request.client.host if request.client else 'unknown',
                },
            )
            # Format error message based on duration
            if rate_limit_seconds < 60:
                wait_message = f'{rate_limit_seconds} seconds'
            elif rate_limit_seconds % 60 == 0:
                wait_message = f'{rate_limit_seconds // 60} minute{"s" if rate_limit_seconds // 60 != 1 else ""}'
            else:
                minutes = rate_limit_seconds // 60
                seconds = rate_limit_seconds % 60
                wait_message = f'{minutes} minute{"s" if minutes != 1 else ""} and {seconds} second{"s" if seconds != 1 else ""}'

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f'Too many requests. Please wait {wait_message} before trying again.',
            )
    except HTTPException:
        # Re-raise HTTPException (rate limit exceeded)
        raise
    except Exception as e:
        # Log error but allow request (fail open) to avoid blocking legitimate users
        logger.warning(f'Error checking rate limit: {e}', exc_info=True)
        return
