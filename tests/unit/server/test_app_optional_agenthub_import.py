from unittest.mock import patch

import pytest

from openhands.server.app import _import_optional_agenthub


def test_import_optional_agenthub_ignores_missing_agenthub():
    error = ModuleNotFoundError("No module named 'openhands.agenthub'")
    error.name = 'openhands.agenthub'
    with patch(
        'openhands.server.app.importlib.import_module',
        side_effect=error,
    ):
        _import_optional_agenthub()


def test_import_optional_agenthub_reraises_other_missing_modules():
    error = ModuleNotFoundError("No module named 'foo'")
    error.name = 'foo'
    with patch(
        'openhands.server.app.importlib.import_module',
        side_effect=error,
    ):
        with pytest.raises(ModuleNotFoundError):
            _import_optional_agenthub()
from unittest.mock import patch

import pytest

from openhands.server.app import _load_optional_agenthub


def test_optional_agenthub_import_is_ignored_when_module_missing():
    missing_agenthub = ModuleNotFoundError("No module named 'openhands.agenthub'")
    missing_agenthub.name = 'openhands.agenthub'
    with patch(
        'openhands.server.app.importlib.import_module',
        side_effect=missing_agenthub,
    ):
        _load_optional_agenthub()


def test_optional_agenthub_import_reraises_other_missing_module_errors():
    missing_dependency = ModuleNotFoundError("No module named 'yaml'")
    missing_dependency.name = 'yaml'
    with patch(
        'openhands.server.app.importlib.import_module',
        side_effect=missing_dependency,
    ):
        with pytest.raises(ModuleNotFoundError):
            _load_optional_agenthub()
