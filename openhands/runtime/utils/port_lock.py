# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
"""File-based port locking system for preventing race conditions in port allocation."""

import os
import random
import socket
import tempfile
import time
from typing import Optional

from openhands.core.logger import openhands_logger as logger

# Import fcntl only on Unix systems
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


class PortLock:
    """File-based lock for a specific port to prevent race conditions."""

    def __init__(self, port: int, lock_dir: Optional[str] = None):
        self.port = port
        self.lock_dir = lock_dir or os.path.join(
            tempfile.gettempdir(), 'openhands_port_locks'
        )
        self.lock_file_path = os.path.join(self.lock_dir, f'port_{port}.lock')
        self.lock_fd: Optional[int] = None
        self._locked = False

        # Ensure lock directory exists
        os.makedirs(self.lock_dir, exist_ok=True)

    def acquire(self, timeout: float = 1.0) -> bool:
        """Acquire the lock for this port.

        Args:
            timeout: Maximum time to wait for the lock

        Returns:
            True if lock was acquired, False otherwise
        """
        if self._locked:
            return True

        try:
            if HAS_FCNTL:
                # Unix-style file locking with fcntl
                self.lock_fd = os.open(
                    self.lock_file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC
                )

                # Try to acquire exclusive lock with timeout
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self._locked = True

                        # Write port number to lock file for debugging
                        os.write(self.lock_fd, f'{self.port}\n'.encode())
                        os.fsync(self.lock_fd)

                        logger.debug(f'Acquired lock for port {self.port}')
                        return True
                    except (OSError, IOError):
                        # Lock is held by another process, wait a bit
                        time.sleep(0.01)

                # Timeout reached
                if self.lock_fd:
                    os.close(self.lock_fd)
                    self.lock_fd = None
                return False
            else:
                # Windows fallback: use atomic file creation
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        # Try to create lock file exclusively
                        self.lock_fd = os.open(
                            self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                        )
                        self._locked = True

                        # Write port number to lock file for debugging
                        os.write(self.lock_fd, f'{self.port}\n'.encode())
                        os.fsync(self.lock_fd)

                        logger.debug(f'Acquired lock for port {self.port}')
                        return True
                    except OSError:
                        # Lock file already exists, wait a bit
                        time.sleep(0.01)

                # Timeout reached
                return False

        except Exception as e:
            logger.debug(f'Failed to acquire lock for port {self.port}: {e}')
            if self.lock_fd:
                try:
                    os.close(self.lock_fd)
                except OSError:
                    pass
                self.lock_fd = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self.lock_fd is not None:
            try:
                if HAS_FCNTL:
                    # Unix: unlock and close
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)

                os.close(self.lock_fd)

                # Remove lock file (both Unix and Windows)
                try:
                    os.unlink(self.lock_file_path)
                except FileNotFoundError:
                    pass
                logger.debug(f'Released lock for port {self.port}')
            except Exception as e:
                logger.warning(f'Error releasing lock for port {self.port}: {e}')
            finally:
                self.lock_fd = None
                self._locked = False

    def __enter__(self) -> 'PortLock':
        if not self.acquire():
            raise OSError(f'Could not acquire lock for port {self.port}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def is_locked(self) -> bool:
        return self._locked


def find_available_port_with_lock(
    min_port: int = 30000,
    max_port: int = 39999,
    max_attempts: int = 20,
    bind_address: str = '0.0.0.0',
    lock_timeout: float = 1.0,
) -> Optional[tuple[int, PortLock]]:
    """Find an available port and acquire a lock for it.

    This function combines file-based locking with port availability checking
    to prevent race conditions in multi-process scenarios.

    Args:
        min_port: Minimum port number to try
        max_port: Maximum port number to try
        max_attempts: Maximum number of ports to try
        bind_address: Address to bind to when checking availability
        lock_timeout: Timeout for acquiring port lock

    Returns:
        Tuple of (port, lock) if successful, None otherwise
    """
    rng = random.SystemRandom()

    # Try random ports first for better distribution
    random_attempts = min(max_attempts // 2, 10)
    for _ in range(random_attempts):
        port = rng.randint(min_port, max_port)

        # Try to acquire lock first
        lock = PortLock(port)
        if lock.acquire(timeout=lock_timeout):
            # Check if port is actually available
            if _check_port_available(port, bind_address):
                logger.debug(f'Found and locked available port {port}')
                return port, lock
            else:
                # Port is locked but not available (maybe in TIME_WAIT state)
                lock.release()

        # Small delay to reduce contention
        time.sleep(0.001)

    # If random attempts failed, try sequential search
    remaining_attempts = max_attempts - random_attempts
    start_port = rng.randint(min_port, max_port - remaining_attempts)

    for i in range(remaining_attempts):
        port = start_port + i
        if port > max_port:
            port = min_port + (port - max_port - 1)

        # Try to acquire lock first
        lock = PortLock(port)
        if lock.acquire(timeout=lock_timeout):
            # Check if port is actually available
            if _check_port_available(port, bind_address):
                logger.debug(f'Found and locked available port {port}')
                return port, lock
            else:
                # Port is locked but not available
                lock.release()

        # Small delay to reduce contention
        time.sleep(0.001)

    logger.error(
        f'Could not find and lock available port in range {min_port}-{max_port} after {max_attempts} attempts'
    )
    return None


def _check_port_available(port: int, bind_address: str = '0.0.0.0') -> bool:
    """Check if a port is available by trying to bind to it."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_address, port))
        sock.close()
        return True
    except OSError:
        return False


def cleanup_stale_locks(max_age_seconds: int = 300) -> int:
    """Clean up stale lock files.

    Args:
        max_age_seconds: Maximum age of lock files before they're considered stale

    Returns:
        Number of lock files cleaned up
    """
    lock_dir = os.path.join(tempfile.gettempdir(), 'openhands_port_locks')
    if not os.path.exists(lock_dir):
        return 0

    cleaned = 0
    current_time = time.time()

    try:
        for filename in os.listdir(lock_dir):
            if filename.startswith('port_') and filename.endswith('.lock'):
                lock_path = os.path.join(lock_dir, filename)
                try:
                    # Check if lock file is old
                    stat = os.stat(lock_path)
                    if current_time - stat.st_mtime > max_age_seconds:
                        # Try to remove stale lock
                        os.unlink(lock_path)
                        cleaned += 1
                        logger.debug(f'Cleaned up stale lock file: {filename}')
                except (OSError, FileNotFoundError):
                    # File might have been removed by another process
                    pass
    except OSError:
        # Directory might not exist or be accessible
        pass

    if cleaned > 0:
        logger.info(f'Cleaned up {cleaned} stale port lock files')

    return cleaned
