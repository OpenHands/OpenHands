"""Tests for sandbox_spec_service helpers.

Covers ``get_agent_server_image`` (derived from the installed
``openhands-agent-server`` package, with a deprecation warning when the legacy
env-var overrides are set) and ``is_custom_sandbox_spec`` (which compares a
sandbox spec id against the bundled default).
"""

import importlib.metadata
import logging
from unittest.mock import patch

import pytest

from openhands.app_server.sandbox import sandbox_spec_service as module
from openhands.app_server.sandbox.sandbox_spec_service import (
    get_agent_server_image,
    is_custom_sandbox_spec,
)


@pytest.fixture(autouse=True)
def _clear_get_agent_server_image_cache():
    """``get_agent_server_image`` is memoized via ``functools.cache``; clear the
    cache around every test so we don't leak env-var state between cases."""
    get_agent_server_image.cache_clear()
    yield
    get_agent_server_image.cache_clear()


def test_get_agent_server_image_derived_from_package_version():
    """The URL must be built from the installed openhands-agent-server version,
    not a hand-maintained constant — that's the whole point of removing the
    drift-prone AGENT_SERVER_IMAGE string."""
    fake_version = '9.9.9'
    with patch.object(
        importlib.metadata,
        'version',
        return_value=fake_version,
    ):
        # cache_clear from the fixture forces a re-derivation inside the patched context.
        get_agent_server_image.cache_clear()
        assert get_agent_server_image() == (
            f'ghcr.io/openhands/agent-server:{fake_version}-python'
        )


def test_get_agent_server_image_returns_consistent_value_within_process():
    """``@cache`` should make repeat calls return the same object without
    re-reading importlib.metadata or re-checking env vars."""
    with patch.object(
        importlib.metadata,
        'version',
        return_value='1.0.0',
    ) as version_mock:
        first = get_agent_server_image()
        # Call many times — version_mock should only be hit once thanks to @cache.
        for _ in range(5):
            assert get_agent_server_image() is first
        assert version_mock.call_count == 1


def test_get_agent_server_image_warns_when_env_vars_set(caplog):
    """A deprecation warning must fire when either legacy env var is set."""
    fake_version = '1.0.0'
    with patch.object(importlib.metadata, 'version', return_value=fake_version):
        get_agent_server_image.cache_clear()
        with caplog.at_level(logging.WARNING, logger=module.__name__):
            with patch.dict(
                'os.environ',
                {'AGENT_SERVER_IMAGE_TAG': '1.31.1-python'},
                clear=False,
            ):
                get_agent_server_image.cache_clear()
                get_agent_server_image()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, 'expected a deprecation warning when AGENT_SERVER_IMAGE_TAG is set'
    assert 'no longer supported' in warnings[0].getMessage()


def test_get_agent_server_image_warning_fires_once_per_process(caplog):
    """The @cache decorator exists specifically to suppress repeated warnings.
    Verify that contract: many calls produce exactly one warning."""
    fake_version = '1.0.0'
    with patch.object(importlib.metadata, 'version', return_value=fake_version):
        get_agent_server_image.cache_clear()
        with caplog.at_level(logging.WARNING, logger=module.__name__):
            with patch.dict(
                'os.environ',
                {'AGENT_SERVER_IMAGE_REPOSITORY': 'example.com/agent-server'},
                clear=False,
            ):
                get_agent_server_image.cache_clear()
                for _ in range(10):
                    get_agent_server_image()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, (
        f'expected exactly one deprecation warning per process, got {len(warnings)}'
    )


def test_get_agent_server_image_no_warning_without_env_vars(caplog):
    """No env vars, no warning — the deprecation must stay silent on clean installs."""
    fake_version = '1.0.0'
    with patch.object(importlib.metadata, 'version', return_value=fake_version):
        get_agent_server_image.cache_clear()
        with caplog.at_level(logging.WARNING, logger=module.__name__):
            with patch.dict('os.environ', {}, clear=True):
                get_agent_server_image.cache_clear()
                get_agent_server_image()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings, 'no deprecation warning should fire on a clean install'


def test_get_agent_server_image_ignores_env_vars():
    """The legacy env vars are no-ops: setting them must not change the URL."""
    fake_version = '1.0.0'
    with patch.object(importlib.metadata, 'version', return_value=fake_version):
        get_agent_server_image.cache_clear()
        with patch.dict(
            'os.environ',
            {
                'AGENT_SERVER_IMAGE_REPOSITORY': 'example.com/agent-server',
                'AGENT_SERVER_IMAGE_TAG': '9.9.9-python',
            },
            clear=False,
        ):
            get_agent_server_image.cache_clear()
            assert get_agent_server_image() == (
                f'ghcr.io/openhands/agent-server:{fake_version}-python'
            )


def test_is_custom_sandbox_spec_false_for_bundled_default():
    """A spec id equal to the bundled default is, by definition, not custom."""
    with patch.object(importlib.metadata, 'version', return_value='1.0.0'):
        get_agent_server_image.cache_clear()
        bundled = get_agent_server_image()
        assert is_custom_sandbox_spec(bundled) is False


def test_is_custom_sandbox_spec_true_for_runtime_api_image():
    """A spec id from runtime-api (any non-default image) must be flagged custom."""
    with patch.object(importlib.metadata, 'version', return_value='1.0.0'):
        get_agent_server_image.cache_clear()
        assert is_custom_sandbox_spec('ghcr.io/some/custom:0.0.1') is True


def test_get_agent_server_image_propagates_package_not_found():
    """openhands-agent-server is a hard runtime dependency; a missing install
    must surface at import time as PackageNotFoundError, not silently degrade."""
    get_agent_server_image.cache_clear()
    with patch.object(
        importlib.metadata,
        'version',
        side_effect=importlib.metadata.PackageNotFoundError('openhands-agent-server'),
    ):
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            get_agent_server_image()
