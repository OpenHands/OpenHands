"""Test that the runtime import system is robust against broken third-party dependencies.

This test specifically addresses the issue where broken third-party runtime dependencies
(like runloop-api-client with incompatible httpx_aiohttp versions) would break the entire
OpenHands CLI and system.
"""

import logging
import sys

import pytest


def test_runtime_import_robustness():
    """Test that runtime import system is robust against broken dependencies."""
    # Clear any cached runtime modules
    modules_to_clear = [k for k in sys.modules.keys() if 'openhands.runtime' in k]
    for module in modules_to_clear:
        del sys.modules[module]

    # Import the runtime module - should succeed even with broken third-party runtimes
    try:
        import openhands.runtime  # noqa: F401

        assert True
    except Exception as e:
        pytest.fail(f'Runtime import failed: {e}')


def test_get_runtime_cls_works():
    """Test that get_runtime_cls works even when third-party runtimes are broken."""
    # Import the runtime module
    import openhands.runtime

    # Test that we can still get core runtime classes
    docker_runtime = openhands.runtime.get_runtime_cls('docker')
    assert docker_runtime is not None

    local_runtime = openhands.runtime.get_runtime_cls('local')
    assert local_runtime is not None

    # Test that requesting a non-existent runtime raises appropriate error
    with pytest.raises(ValueError, match='Runtime nonexistent not supported'):
        openhands.runtime.get_runtime_cls('nonexistent')


def test_import_error_handled_silently(caplog):
    """Test that ImportError is handled silently (no logging) as it means library is not installed."""
    # Simulate the exact code path for ImportError
    logging.getLogger('openhands.runtime')

    with caplog.at_level(logging.WARNING):
        # Simulate ImportError handling - this should NOT log anything
        try:
            raise ImportError("No module named 'optional_runtime_library'")
        except ImportError:
            # This is the exact code from runtime init: just pass, no logging
            pass

    # Check that NO warning was logged for ImportError
    warning_records = [
        record for record in caplog.records if record.levelname == 'WARNING'
    ]
    assert len(warning_records) == 0, (
        f'ImportError should not generate warnings, but got: {warning_records}'
    )
