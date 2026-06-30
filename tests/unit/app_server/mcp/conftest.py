"""Shared fixtures for MCP app-server unit tests."""

from __future__ import annotations

import os

import pytest

_ALLOW_SHORT_CONTEXT_WINDOWS = 'ALLOW_SHORT_CONTEXT_WINDOWS'


@pytest.fixture(autouse=True)
def allow_short_context_windows():
    """Allow small context windows so tests can construct LLM with gpt-4 etc."""
    old = os.environ.pop(_ALLOW_SHORT_CONTEXT_WINDOWS, None)
    os.environ[_ALLOW_SHORT_CONTEXT_WINDOWS] = 'true'
    try:
        yield
    finally:
        if old is not None:
            os.environ[_ALLOW_SHORT_CONTEXT_WINDOWS] = old
        else:
            os.environ.pop(_ALLOW_SHORT_CONTEXT_WINDOWS, None)
