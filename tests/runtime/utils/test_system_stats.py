"""Tests for system stats utilities."""

import time
from unittest.mock import patch

import psutil

from openhands.app_server.status.system_stats import (
    get_system_info,
    get_system_stats,
    update_last_execution_time,
)


def test_get_system_stats():
    """Test that get_system_stats returns valid system statistics."""
    stats = get_system_stats()

    # Test structure
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {'cpu_percent', 'memory', 'disk', 'io'}

    # Test CPU stats
    assert isinstance(stats['cpu_percent'], float)
    assert 0 <= stats['cpu_percent'] <= 100 * psutil.cpu_count()

    # Test memory stats
    assert isinstance(stats['memory'], dict)
    assert set(stats['memory'].keys()) == {'rss', 'vms', 'percent'}
    assert isinstance(stats['memory']['rss'], int)
    assert isinstance(stats['memory']['vms'], int)
    assert isinstance(stats['memory']['percent'], float)
    assert stats['memory']['rss'] > 0
    assert stats['memory']['vms'] > 0
    assert 0 <= stats['memory']['percent'] <= 100

    # Test disk stats
    assert isinstance(stats['disk'], dict)
    assert set(stats['disk'].keys()) == {'total', 'used', 'free', 'percent'}
    assert isinstance(stats['disk']['total'], int)
    assert isinstance(stats['disk']['used'], int)
    assert isinstance(stats['disk']['free'], int)
    assert isinstance(stats['disk']['percent'], float)
    assert stats['disk']['total'] > 0
    assert stats['disk']['used'] >= 0
    assert stats['disk']['free'] >= 0
    assert 0 <= stats['disk']['percent'] <= 100
    # Verify that used + free is less than or equal to total
    # (might not be exactly equal due to filesystem overhead)
    assert stats['disk']['used'] + stats['disk']['free'] <= stats['disk']['total']

    # Test I/O stats
    assert isinstance(stats['io'], dict)
    assert set(stats['io'].keys()) == {'read_bytes', 'write_bytes'}
    assert isinstance(stats['io']['read_bytes'], int)
    assert isinstance(stats['io']['write_bytes'], int)
    assert stats['io']['read_bytes'] >= 0
    assert stats['io']['write_bytes'] >= 0


def test_get_system_stats_stability():
    """Test that get_system_stats can be called multiple times without errors."""
    # Call multiple times to ensure stability
    for _ in range(3):
        stats = get_system_stats()
        assert isinstance(stats, dict)
        assert stats['cpu_percent'] >= 0


def test_get_system_info():
    """Test that get_system_info returns valid system information."""
    with patch(
        'openhands.runtime.utils.system_stats.get_system_stats'
    ) as mock_get_stats:
        mock_get_stats.return_value = {'cpu_percent': 10.0}

        info = get_system_info()

        # Test structure
        assert isinstance(info, dict)
        assert set(info.keys()) == {'uptime', 'idle_time', 'resources'}

        # Test values
        assert isinstance(info['uptime'], float)
        assert isinstance(info['idle_time'], float)
        assert info['uptime'] > 0
        assert info['idle_time'] >= 0
        assert info['resources'] == {'cpu_percent': 10.0}

        # Verify get_system_stats was called
        mock_get_stats.assert_called_once()


def test_update_last_execution_time():
    """Test that update_last_execution_time updates the last execution time."""
    # Get initial system info
    initial_info = get_system_info()
    initial_idle_time = initial_info['idle_time']

    # Wait a bit to ensure time difference
    time.sleep(0.1)

    # Update last execution time
    update_last_execution_time()

    # Get updated system info
    updated_info = get_system_info()
    updated_idle_time = updated_info['idle_time']

    # The idle time should be reset (close to zero)
    assert updated_idle_time < initial_idle_time
    assert updated_idle_time < 0.1  # Should be very small


def test_idle_time_increases_without_updates():
    """Test that idle_time increases when no updates are made."""
    # Update last execution time to reset idle time
    update_last_execution_time()

    # Get initial system info
    initial_info = get_system_info()
    initial_idle_time = initial_info['idle_time']

    # Wait a bit
    time.sleep(0.2)

    # Get updated system info without calling update_last_execution_time
    updated_info = get_system_info()
    updated_idle_time = updated_info['idle_time']

    # The idle time should have increased
    assert updated_idle_time > initial_idle_time
    assert updated_idle_time >= 0.2  # Should be at least the sleep time


@patch('time.time')
def test_idle_time_calculation(mock_time):
    """Test that idle_time is calculated correctly."""
    # Mock time.time() to return controlled values
    mock_time.side_effect = [
        100.0,  # Initial _start_time
        100.0,  # Initial _last_execution_time
        110.0,  # Current time in get_system_info
    ]

    # Import the module again to reset the global variables with our mocked time
    import importlib

    import openhands.app_server.status.system_stats

    importlib.reload(openhands.app_server.status.system_stats)

    # Get system info
    from openhands.app_server.status.system_stats import get_system_info

    info = get_system_info()

    # Verify idle_time calculation
    assert info['uptime'] == 10.0  # 110 - 100
    assert info['idle_time'] == 10.0  # 110 - 100
