from unittest.mock import patch

import pytest

from openhands.server.optional_legacy_imports import (
    _import_optional_agenthub,
    _load_optional_security_api_router,
)


def test_import_optional_agenthub_ignores_missing_agenthub():
    error = ModuleNotFoundError("No module named 'openhands.agenthub'")
    error.name = "openhands.agenthub"
    with patch(
        "openhands.server.optional_legacy_imports.importlib.import_module",
        side_effect=error,
    ):
        _import_optional_agenthub()


def test_import_optional_agenthub_reraises_other_missing_modules():
    error = ModuleNotFoundError("No module named 'foo'")
    error.name = "foo"
    with patch(
        "openhands.server.optional_legacy_imports.importlib.import_module",
        side_effect=error,
    ):
        with pytest.raises(ModuleNotFoundError):
            _import_optional_agenthub()


def test_load_optional_security_router_returns_none_when_module_missing():
    missing_security = ModuleNotFoundError(
        "No module named 'openhands.server.routes.security'"
    )
    missing_security.name = "openhands.server.routes.security"
    with patch(
        "openhands.server.optional_legacy_imports.importlib.import_module",
        side_effect=missing_security,
    ):
        assert _load_optional_security_api_router() is None


def test_load_optional_security_router_reraises_other_missing_module_errors():
    missing_dependency = ModuleNotFoundError("No module named 'yaml'")
    missing_dependency.name = "yaml"
    with patch(
        "openhands.server.optional_legacy_imports.importlib.import_module",
        side_effect=missing_dependency,
    ):
        with pytest.raises(ModuleNotFoundError):
            _load_optional_security_api_router()
