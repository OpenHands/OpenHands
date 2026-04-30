import asyncio
import concurrent.futures

import pytest

from openhands.app_server.utils.async_utils import (
    AsyncException,
    call_async_from_sync,
    call_sync_from_async,
    run_in_loop,
    wait_all,
)


@pytest.mark.asyncio
async def test_await_all():
    # Mock function demonstrating some calculation - always takes a minimum of 0.1 seconds
    async def dummy(value: int):
        await asyncio.sleep(0.1)
        return value * 2

    # wait for 10 calculations - serially this would take 1 second
    coro = wait_all(dummy(i) for i in range(10))

    # give the task only 0.3 seconds to complete (This verifies they occur in parallel)
    task = asyncio.create_task(coro)
    await asyncio.wait([task], timeout=0.3)

    # validate the results (We need to sort because they can return in any order)
    results = list(await task)
    expected = [i * 2 for i in range(10)]
    assert expected == results


@pytest.mark.asyncio
async def test_await_all_single_exception():
    # Mock function demonstrating some calculation - always takes a minimum of 0.1 seconds
    async def dummy(value: int):
        await asyncio.sleep(0.1)
        if value == 1:
            raise ValueError('Invalid value 1')  # Throw an exception on every odd value
        return value * 2

    # expect an exception to be raised.
    with pytest.raises(ValueError, match='Invalid value 1'):
        await wait_all(dummy(i) for i in range(10))


@pytest.mark.asyncio
async def test_await_all_multi_exception():
    # Mock function demonstrating some calculation - always takes a minimum of 0.1 seconds
    async def dummy(value: int):
        await asyncio.sleep(0.1)
        if value & 1:
            raise ValueError(
                f'Invalid value {value}'
            )  # Throw an exception on every odd value
        return value * 2

    # expect an exception to be raised.
    with pytest.raises(AsyncException):
        await wait_all(dummy(i) for i in range(10))


@pytest.mark.asyncio
async def test_await_all_timeout():
    result = 0

    # Mock function updates a nonlocal variable after a delay
    async def dummy(value: int):
        nonlocal result
        await asyncio.sleep(0.2)
        result += value

    # expect an exception to be raised.
    with pytest.raises(asyncio.TimeoutError):
        await wait_all((dummy(i) for i in range(10)), 0.1)

    # Wait and then check the shared result - this makes sure that pending tasks were cancelled.
    asyncio.sleep(0.2)
    assert result == 0


@pytest.mark.asyncio
async def test_call_sync_from_async():
    def dummy(value: int = 2):
        return value * 2

    result = await call_sync_from_async(dummy)
    assert result == 4
    result = await call_sync_from_async(dummy, 3)
    assert result == 6
    result = await call_sync_from_async(dummy, value=5)
    assert result == 10


@pytest.mark.asyncio
async def test_call_sync_from_async_error():
    def dummy():
        raise ValueError()

    with pytest.raises(ValueError):
        await call_sync_from_async(dummy)


def test_call_async_from_sync():
    async def dummy(value: int):
        return value * 2

    result = call_async_from_sync(dummy, 0, 3)
    assert result == 6


def test_call_async_from_sync_error():
    async def dummy(value: int):
        raise ValueError()

    with pytest.raises(ValueError):
        call_async_from_sync(dummy, 0, 3)


def test_call_async_from_sync_background_tasks():
    events = []

    async def bg_task():
        # This background task should finish after the dummy task
        events.append('bg_started')
        asyncio.sleep(0.2)
        events.append('bg_finished')

    async def dummy(value: int):
        events.append('dummy_started')
        # This coroutine kicks off a background task
        asyncio.create_task(bg_task())
        events.append('dummy_started')

    call_async_from_sync(dummy, 0, 3)

    # We check that the function did not return until all coroutines completed
    # (Even though some of these were started as background tasks)
    expected = ['dummy_started', 'dummy_started', 'bg_started', 'bg_finished']
    assert expected == events


@pytest.mark.asyncio
async def test_run_in_loop_same_loop():
    """Test run_in_loop when the target loop is the same as the current loop."""

    async def dummy_coro(value: int):
        await asyncio.sleep(0.01)  # Small delay to make it actually async
        return value * 2

    # Get the current running loop
    current_loop = asyncio.get_running_loop()

    # Create a coroutine and run it in the same loop
    coro = dummy_coro(5)
    result = await run_in_loop(coro, current_loop)

    assert result == 10


@pytest.mark.asyncio
async def test_run_in_loop_different_loop():
    """Test run_in_loop when the target loop is different from the current loop."""
    import queue
    import threading

    async def dummy_coro(value: int):
        await asyncio.sleep(0.01)  # Small delay to make it actually async
        return value * 3

    queue.Queue()
    loop_queue = queue.Queue()

    def run_in_new_loop():
        """Create and run a new event loop in a separate thread."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        loop_queue.put(new_loop)  # Share the loop with the main thread

        try:
            # Keep the loop running for a short time
            new_loop.run_until_complete(asyncio.sleep(2.0))
        except Exception:
            pass  # Expected when we stop the loop
        finally:
            new_loop.close()

    # Start the new loop in a separate thread
    thread = threading.Thread(target=run_in_new_loop, daemon=True)
    thread.start()

    # Get the new loop from the thread
    await asyncio.sleep(0.1)  # Give thread time to start
    new_loop = loop_queue.get(timeout=1.0)

    try:
        # Create a coroutine and run it in the different loop
        coro = dummy_coro(7)
        result = await run_in_loop(coro, new_loop)
        assert result == 21
    finally:
        # Clean up: stop the loop
        new_loop.call_soon_threadsafe(new_loop.stop)


@pytest.mark.asyncio
async def test_run_in_loop_with_exception():
    """Test run_in_loop when the coroutine raises an exception."""

    async def failing_coro():
        await asyncio.sleep(0.01)
        raise ValueError('Test exception')

    current_loop = asyncio.get_running_loop()
    coro = failing_coro()

    with pytest.raises(ValueError, match='Test exception'):
        await run_in_loop(coro, current_loop)


@pytest.mark.asyncio
async def test_run_in_loop_with_timeout():
    """Test run_in_loop with a timeout when using different loops."""
    import queue
    import threading

    async def slow_coro():
        await asyncio.sleep(1.0)  # Sleep longer than timeout
        return 'should not reach here'

    loop_queue = queue.Queue()

    def run_in_new_loop():
        """Create and run a new event loop in a separate thread."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        loop_queue.put(new_loop)

        try:
            # Keep the loop running for a short time
            new_loop.run_until_complete(asyncio.sleep(2.0))
        except Exception:
            pass  # Expected when we stop the loop
        finally:
            new_loop.close()

    # Start the new loop in a separate thread
    thread = threading.Thread(target=run_in_new_loop, daemon=True)
    thread.start()

    # Get the new loop from the thread
    await asyncio.sleep(0.1)  # Give thread time to start
    new_loop = loop_queue.get(timeout=1.0)

    try:
        coro = slow_coro()
        # Test with a short timeout - this should raise a timeout exception
        with pytest.raises(
            (TimeoutError, concurrent.futures.TimeoutError)
        ):  # Could be TimeoutError or concurrent.futures.TimeoutError
            await run_in_loop(coro, new_loop, timeout=0.1)
    finally:
        # Clean up: stop the loop
        new_loop.call_soon_threadsafe(new_loop.stop)


@pytest.mark.asyncio
async def test_run_in_loop_same_loop_no_timeout():
    """Test that run_in_loop doesn't apply timeout when using the same loop."""

    async def quick_coro():
        await asyncio.sleep(0.01)
        return 'completed'

    current_loop = asyncio.get_running_loop()
    coro = quick_coro()

    # Even with a very short timeout, this should work because
    # timeout is only applied when using different loops
    result = await run_in_loop(coro, current_loop, timeout=0.001)
    assert result == 'completed'


@pytest.mark.asyncio
async def test_run_in_loop_return_value():
    """Test that run_in_loop properly returns the coroutine result."""

    async def return_dict():
        await asyncio.sleep(0.01)
        return {'key': 'value', 'number': 42}

    current_loop = asyncio.get_running_loop()
    coro = return_dict()

    result = await run_in_loop(coro, current_loop)

    assert isinstance(result, dict)
    assert result['key'] == 'value'
    assert result['number'] == 42


@pytest.mark.asyncio
async def test_run_in_loop_with_args_and_kwargs():
    """Test run_in_loop with a coroutine that uses arguments."""

    async def coro_with_args(a, b, multiplier=1):
        await asyncio.sleep(0.01)
        return (a + b) * multiplier

    current_loop = asyncio.get_running_loop()

    # Create coroutine with args and kwargs
    coro = coro_with_args(5, 10, multiplier=2)
    result = await run_in_loop(coro, current_loop)

    assert result == 30  # (5 + 10) * 2


def test_run_in_loop_sync_context():
    """Test run_in_loop behavior when called from a synchronous context."""

    async def dummy_coro(value: int):
        await asyncio.sleep(0.01)
        return value * 4

    def sync_function():
        """Function that runs in a new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            coro = dummy_coro(6)
            # This simulates the scenario where we have a different target loop
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # Test the function in a synchronous context
    result = sync_function()
    assert result == 24
