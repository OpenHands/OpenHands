"""Usage:

Call setup_rate_limit_handler on your FastAPI app to add the exception handler

Create a rate limiter like:
    `rate_limiter = create_redis_rate_limiter("10/second; 100/minute")`

Call hit() with some key and allow the RateLimitException to propagate:
    `rate_limiter.hit('some action', user_id)`
"""

import time
from dataclasses import dataclass

import limits
from fastapi.responses import JSONResponse
from starlette.applications import Request, Response, Starlette
from starlette.exceptions import HTTPException
from storage.redis import get_redis_authed_url

from openhands.core.logger import openhands_logger as logger


def setup_rate_limit_handler(app: Starlette):
    """Add exception handler that
    """
    app.add_exception_handler(RateLimitException, _rate_limit_exceeded_handler)


@dataclass
class RateLimitResult:
    """Result of a rate limit check, times in seconds"""

    description: str
    remaining: int
    reset_time: int
    retry_after: int | None = None

    def add_headers(self, response: Response) -> None:
        """Add rate limit headers to a response"""
        response.headers['X-RateLimit-Limit'] = self.description
        response.headers['X-RateLimit-Remaining'] = str(self.remaining)
        response.headers['X-RateLimit-Reset'] = str(self.reset_time)
        if self.retry_after is not None:
            response.headers['Retry-After'] = str(self.retry_after)


class RateLimiter:
    strategy: limits.aio.strategies.RateLimiter
    limit_items: list[limits.RateLimitItem]

    def __init__(self, strategy: limits.aio.strategies.RateLimiter, windows: str):
        self.strategy = strategy
        self.limit_items = limits.parse_many(windows)

    async def hit(self, namespace: str, key: str):
        """Raises RateLimitException when limit is hit.
        Logs and swallows exceptions and logs if lookup fails.
        """
        for lim in self.limit_items:
            allowed = True
            try:
                allowed = await self.strategy.hit(lim, namespace, key)
            except Exception:
                logger.exception('Rate limit check could not complete, redis issue?')
            if not allowed:
                logger.info(f'Rate limit hit for {namespace}:{key}')
                try:
                    result = await self._get_stats_as_result(lim, namespace, key)
                except Exception:
                    logger.exception(
                        'Rate limit exceeded but window lookup failed, swallowing'
                    )
                else:
                    raise RateLimitException(result)

    async def _get_stats_as_result(
        self, lim: limits.RateLimitItem, namespace: str, key: str
    ) -> RateLimitResult:
        """Lookup rate limit window stats and return a RateLimitResult with the data needed for response headers.
        """
        stats: limits.WindowStats = await self.strategy.get_window_stats(
            lim, namespace, key
        )
        return RateLimitResult(
            description=str(lim),
            remaining=stats.remaining,
            reset_time=int(stats.reset_time),
            retry_after=int(stats.reset_time - time.time())
            if stats.remaining == 0
            else None,
        )


def create_redis_rate_limiter(windows: str) -> RateLimiter:
    """Create a RateLimiter with the Redis backend and "Fixed Window" strategy.
    windows arg example: "10/second; 100/minute"
    """
    backend = limits.aio.storage.RedisStorage(f'async+{get_redis_authed_url()}')
    strategy = limits.aio.strategies.FixedWindowRateLimiter(backend)
    return RateLimiter(strategy, windows)


class RateLimitException(HTTPException):
    """exception raised when a rate limit is hit.
    """

    result: RateLimitResult

    def __init__(self, result: RateLimitResult) -> None:
        self.result = result
        super(RateLimitException, self).__init__(
            status_code=429, detail=result.description
        )


def _rate_limit_exceeded_handler(request: Request, exc: Exception) -> Response:
    """Build a simple JSON response that includes the details of the rate limit that was hit.
    """
    logger.info(exc.__class__.__name__)
    if isinstance(exc, RateLimitException):
        response = JSONResponse(
            {'error': f'Rate limit exceeded: { exc.detail}'}, status_code=429
        )
        if exc.result:
            exc.result.add_headers(response)
    else:
        # Shouldn't happen, this handler is only bound to RateLimitException
        response = JSONResponse({'error': 'Rate limit exceeded'}, status_code=429)
    return response
