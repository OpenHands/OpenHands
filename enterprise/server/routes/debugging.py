import asyncio
import os
import time
from threading import Thread

from fastapi import APIRouter, FastAPI
from sqlalchemy import func, select
from storage.database import a_session_maker, get_engine, session_maker
from storage.user import User

from openhands.core.logger import openhands_logger as logger
from openhands.utils.async_utils import wait_all

# Safety flag to prevent chaos routes from being added in production environments
# Only enables these routes in non-production environments
ADD_DEBUGGING_ROUTES = os.environ.get('ADD_DEBUGGING_ROUTES') in ('1', 'true')


def add_debugging_routes(api: FastAPI):
    """
    # HERE BE DRAGONS!
    Chaos scripts for debugging and stress testing the system.

    This module contains endpoints that deliberately stress test and potentially break
    the system to help identify weaknesses and bottlenecks. It includes a safety check
    to ensure these routes are never deployed to production environments.

    The routes in this module are specifically designed for:
    - Testing connection pool behavior under load
    - Simulating database connection exhaustion
    - Testing async vs sync database access patterns
    - Simulating event loop blocking
    """

    if not ADD_DEBUGGING_ROUTES:
        return

    chaos_router = APIRouter(prefix='/debugging')

    @chaos_router.get('/pool-stats')
    def pool_stats() -> dict[str, int]:
        """
        Returns current database connection pool statistics.

        This endpoint provides real-time metrics about the SQLAlchemy connection pool:
        - checked_in: Number of connections currently available in the pool
        - checked_out: Number of connections currently in use
        - overflow: Number of overflow connections created beyond pool_size
        """
        engine = get_engine()
        return {
            'checked_in': engine.pool.checkedin(),
            'checked_out': engine.pool.checkedout(),
            'overflow': engine.pool.overflow(),
        }

    @chaos_router.get('/test-db')
    def test_db(num_tests: int = 10, delay: int = 1) -> str:
        """
        Stress tests the database connection pool using multiple threads.

        Creates multiple threads that each open a database connection, perform a query,
        hold the connection for the specified delay, and then release it.

        Parameters:
            num_tests: Number of concurrent database connections to create
            delay: Number of seconds each connection is held open

        This test helps identify connection pool exhaustion issues and connection
        leaks under concurrent load.
        """
        threads = [Thread(target=_db_check, args=(delay,)) for _ in range(num_tests)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return 'success'

    @chaos_router.get('/a-test-db')
    async def a_chaos_monkey(num_tests: int = 10, delay: int = 1) -> str:
        """
        Stress tests the async database connection pool.

        Similar to /test-db but uses async connections and coroutines instead of threads.
        This endpoint helps compare the behavior of async vs sync connection pools
        under similar load conditions.

        Parameters:
            num_tests: Number of concurrent async database connections to create
            delay: Number of seconds each connection is held open
        """
        await wait_all((_a_db_check(delay) for _ in range(num_tests)))
        return 'success'

    @chaos_router.get('/lock-main-runloop')
    async def lock_main_runloop(duration: int = 10) -> str:
        """
        Deliberately blocks the main asyncio event loop.

        This endpoint uses a synchronous sleep operation in an async function,
        which blocks the entire FastAPI server's event loop for the specified duration.
        This simulates what happens when CPU-intensive operations or blocking I/O
        operations are incorrectly used in async code.

        Parameters:
            duration: Number of seconds to block the event loop

        WARNING: This will make the entire server unresponsive for the duration!
        """
        time.sleep(duration)
        return 'success'

    api.include_router(chaos_router)  # Add routes for readiness checks


def _db_check(delay: int):
    """
    Executes a single request against the database with an artificial delay.

    This helper function:
    1. Opens a database connection from the pool
    2. Executes a simple query to count users
    3. Holds the connection for the specified delay
    4. Logs connection pool statistics
    5. Implicitly returns the connection to the pool when the session closes

    Args:
        delay: Number of seconds to hold the database connection
    """
    with session_maker() as session:
        num_users = session.query(User).count()
        time.sleep(delay)
        engine = get_engine()
        logger.info(
            'check',
            extra={
                'num_users': num_users,
                'checked_in': engine.pool.checkedin(),
                'checked_out': engine.pool.checkedout(),
                'overflow': engine.pool.overflow(),
            },
        )


async def _a_db_check(delay: int):
    """
    Executes a single async request against the database with an artificial delay.

    This is the async version of _db_check that:
    1. Opens an async database connection from the pool
    2. Executes a simple query to count users using SQLAlchemy's async API
    3. Holds the connection for the specified delay using asyncio.sleep
    4. Logs the results
    5. Implicitly returns the connection to the pool when the async session closes

    Args:
        delay: Number of seconds to hold the database connection
    """
    async with a_session_maker() as a_session:
        stmt = select(func.count(User.id))
        num_users = await a_session.execute(stmt)
        await asyncio.sleep(delay)
        logger.info(f'a_num_users:{num_users.scalar_one()}')
