# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
"""Utilities for getting system resource statistics."""

import time

import psutil

_start_time = time.time()
_last_execution_time = time.time()


def get_system_info() -> dict[str, object]:
    current_time = time.time()
    uptime = current_time - _start_time
    idle_time = current_time - _last_execution_time
    return {
        'uptime': uptime,
        'idle_time': idle_time,
        'resources': get_system_stats(),
    }


def update_last_execution_time():
    global _last_execution_time
    _last_execution_time = time.time()


def get_system_stats() -> dict[str, object]:
    """Get current system resource statistics.

    Returns:
        dict: A dictionary containing:
            - cpu_percent: CPU usage percentage for the current process
            - memory: Memory usage stats (rss, vms, percent)
            - disk: Disk usage stats (total, used, free, percent)
            - io: I/O statistics (read/write bytes)
    """
    process = psutil.Process()
    # Get initial CPU percentage (this will return 0.0)
    process.cpu_percent()
    # Wait a bit and get the actual CPU percentage
    time.sleep(0.1)

    with process.oneshot():
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

    disk_usage = psutil.disk_usage('/')

    # Get I/O stats directly from /proc/[pid]/io to avoid psutil's field name assumptions
    try:
        with open(f'/proc/{process.pid}/io', 'rb') as f:
            io_stats = {}
            for line in f:
                if line:
                    try:
                        name, value = line.strip().split(b': ')
                        io_stats[name.decode('ascii')] = int(value)
                    except (ValueError, UnicodeDecodeError):
                        continue
    except (FileNotFoundError, PermissionError):
        io_stats = {'read_bytes': 0, 'write_bytes': 0}

    return {
        'cpu_percent': cpu_percent,
        'memory': {
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'percent': memory_percent,
        },
        'disk': {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': disk_usage.percent,
        },
        'io': {
            'read_bytes': io_stats.get('read_bytes', 0),
            'write_bytes': io_stats.get('write_bytes', 0),
        },
    }
