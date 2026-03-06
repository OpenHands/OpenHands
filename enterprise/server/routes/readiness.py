from fastapi import APIRouter, HTTPException, status
from sqlalchemy.sql import text
from storage.database import a_session_maker
from storage.redis import create_redis_client

from openhands.core.logger import openhands_logger as logger

readiness_router = APIRouter()


@readiness_router.get('/ready')
async def is_ready():
    # Check database connection
    try:
        async with a_session_maker() as session:
            await session.execute(text('SELECT 1'))
    except Exception as e:
        logger.error(f'Database check failed: {str(e)}')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'Database is not accessible: {str(e)}',
        )

    # Check Redis connection
    try:
        redis_client = create_redis_client()
        redis_client.ping()
    except Exception as e:
        logger.error(f'Redis check failed: {str(e)}')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'Redis cache is not accessible: {str(e)}',
        )

    return 'OK'
