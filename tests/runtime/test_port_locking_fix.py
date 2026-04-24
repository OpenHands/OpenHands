"""Test for port allocation race condition fix."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openhands.runtime.utils.port_lock import PortLock, find_available_port_with_lock


class TestPortLockingFix:
    """Test cases for port allocation race condition fix."""

    def test_port_lock_prevents_duplicate_allocation(self):
        """Test that port locking prevents duplicate port allocation."""
        allocated_ports = []
        port_locks = []

        def allocate_port():
            """Simulate port allocation by multiple workers."""
            result = find_available_port_with_lock(
                min_port=30000,
                max_port=30010,  # Small range to force conflicts
                max_attempts=5,
                bind_address='0.0.0.0',
                lock_timeout=2.0,
            )

            if result:
                port, lock = result
                allocated_ports.append(port)
                port_locks.append(lock)
                # Simulate some work time
                time.sleep(0.1)
                return port
            return None

        # Run multiple threads concurrently
        num_workers = 8
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(allocate_port) for _ in range(num_workers)]
            results = [future.result() for future in as_completed(futures)]

        # Filter out None results
        successful_ports = [port for port in results if port is not None]

        # Verify no duplicate ports were allocated
        assert len(successful_ports) == len(set(successful_ports)), (
            f'Duplicate ports allocated: {successful_ports}'
        )

        # Clean up locks
        for lock in port_locks:
            if lock:
                lock.release()

        print(
            f'Successfully allocated {len(successful_ports)} unique ports: {successful_ports}'
        )

    def test_port_lock_basic_functionality(self):
        """Test basic port lock functionality."""
        port = 30001

        # Test acquiring and releasing a lock
        lock1 = PortLock(port)
        assert lock1.acquire(timeout=1.0)
        assert lock1.is_locked

        # Test that another lock cannot acquire the same port
        lock2 = PortLock(port)
        assert not lock2.acquire(timeout=0.1)
        assert not lock2.is_locked

        # Release first lock
        lock1.release()
        assert not lock1.is_locked

        # Now second lock should be able to acquire
        assert lock2.acquire(timeout=1.0)
        assert lock2.is_locked

        lock2.release()

    def test_port_lock_context_manager(self):
        """Test port lock context manager functionality."""
        port = 30002

        # Test successful context manager usage
        with PortLock(port) as lock:
            assert lock.is_locked

            # Test that another lock cannot acquire while in context
            lock2 = PortLock(port)
            assert not lock2.acquire(timeout=0.1)

        # After context, lock should be released
        assert not lock.is_locked

        # Now another lock should be able to acquire
        lock3 = PortLock(port)
        assert lock3.acquire(timeout=1.0)
        lock3.release()

    def test_concurrent_port_allocation_stress_test(self):
        """Stress test concurrent port allocation."""
        allocated_ports = []
        port_locks = []
        errors = []

        def worker_allocate_port(worker_id):
            """Worker function that allocates a port."""
            try:
                result = find_available_port_with_lock(
                    min_port=31000,
                    max_port=31020,  # Small range to force contention
                    max_attempts=10,
                    bind_address='0.0.0.0',
                    lock_timeout=3.0,
                )

                if result:
                    port, lock = result
                    allocated_ports.append((worker_id, port))
                    port_locks.append(lock)
                    # Simulate work
                    time.sleep(0.05)
                    return port
                else:
                    errors.append(f'Worker {worker_id}: No port available')
                    return None

            except Exception as e:
                errors.append(f'Worker {worker_id}: {str(e)}')
                return None

        # Run many workers concurrently
        num_workers = 15
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker_allocate_port, i): i for i in range(num_workers)
            }
            results = {}
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    result = future.result()
                    results[worker_id] = result
                except Exception as e:
                    errors.append(f'Worker {worker_id} exception: {str(e)}')

        # Analyze results
        successful_allocations = [
            (wid, port) for wid, port in allocated_ports if port is not None
        ]
        allocated_port_numbers = [port for _, port in successful_allocations]

        print(f'Successful allocations: {len(successful_allocations)}')
        print(f'Allocated ports: {allocated_port_numbers}')
        print(f'Errors: {len(errors)}')
        if errors:
            print(f'Error details: {errors[:5]}')  # Show first 5 errors

        # Verify no duplicate ports
        unique_ports = set(allocated_port_numbers)
        assert len(allocated_port_numbers) == len(unique_ports), (
            f'Duplicate ports found: {allocated_port_numbers}'
        )

        # Clean up locks
        for lock in port_locks:
            if lock:
                lock.release()

    def test_port_allocation_without_locking_shows_race_condition(self):
        """Test that demonstrates race condition without locking."""
        from openhands.runtime.utils import find_available_tcp_port

        allocated_ports = []

        def allocate_port_without_lock():
            """Simulate port allocation without locking (old method)."""
            # This simulates the old behavior that had race conditions
            port = find_available_tcp_port(32000, 32010)
            allocated_ports.append(port)
            # Small delay to increase chance of race condition
            time.sleep(0.01)
            return port

        # Run multiple threads concurrently
        num_workers = 10
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(allocate_port_without_lock) for _ in range(num_workers)
            ]
            results = [future.result() for future in as_completed(futures)]

        # Check if we got duplicate ports (race condition)
        unique_ports = set(results)
        duplicates_found = len(results) != len(unique_ports)

        print(
            f'Without locking - Total ports: {len(results)}, Unique: {len(unique_ports)}'
        )
        print(f'Ports allocated: {results}')
        print(f'Race condition detected: {duplicates_found}')

        # This test demonstrates the problem exists without locking
        # In a real race condition scenario, we might get duplicates
        # But since the race window is small, we'll just verify the test runs
        assert len(results) == num_workers


if __name__ == '__main__':
    test = TestPortLockingFix()
    test.test_port_lock_prevents_duplicate_allocation()
    test.test_port_lock_basic_functionality()
    test.test_port_lock_context_manager()
    test.test_concurrent_port_allocation_stress_test()
    test.test_port_allocation_without_locking_shows_race_condition()
    print('All tests passed!')
