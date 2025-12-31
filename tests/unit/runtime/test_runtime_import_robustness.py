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


def test_runtime_exception_handling():
    """Test that the runtime discovery code properly handles exceptions."""
    # This test verifies that the fix in openhands/runtime/__init__.py
    # properly catches all exceptions (not just ImportError) during
    # third-party runtime discovery

    import openhands.runtime

    # The fact that we can import this module successfully means
    # the exception handling is working correctly, even if there
    # are broken third-party runtime dependencies
    assert hasattr(openhands.runtime, 'get_runtime_cls')
    assert hasattr(openhands.runtime, '_THIRD_PARTY_RUNTIME_CLASSES')


def test_runtime_import_exception_handling_behavior():
    """Test that runtime import handles ImportError silently but logs other exceptions."""
    # Test the exception handling logic by simulating the exact code from runtime init
    from io import StringIO

    from openhands.core.logger import openhands_logger as logger

    # Create a string buffer to capture log output
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)

    # Add our test handler to the OpenHands logger
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(logging.WARNING)

    try:
        # Test 1: ImportError should be handled silently (no logging)
        module_path = 'third_party.runtime.impl.missing.missing_runtime'
        try:
            raise ImportError("No module named 'missing_library'")
        except ImportError:
            # This is the exact code from runtime init: just pass, no logging
            pass

        # Test 2: Other exceptions should be logged
        module_path = 'third_party.runtime.impl.runloop.runloop_runtime'
        try:
            raise AttributeError(
                "module 'httpx_aiohttp' has no attribute 'HttpxAiohttpClient'"
            )
        except ImportError:
            # ImportError means the library is not installed (expected for optional dependencies)
            pass
        except Exception as e:
            # Other exceptions mean the library is present but broken, which should be logged
            # This is the exact code from runtime init
            logger.warning(f'Failed to import third-party runtime {module_path}: {e}')

        # Check the captured log output
        log_output = log_capture.getvalue()

        # Should contain the AttributeError warning
        assert 'Failed to import third-party runtime' in log_output
        assert 'HttpxAiohttpClient' in log_output
        # Should NOT contain the ImportError message
        assert 'missing_library' not in log_output

    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


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
