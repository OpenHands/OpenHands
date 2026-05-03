"""Import hooks for the legacy ``openhands.server.app`` compatibility shim.

Keeping this module tiny avoids pulling ``openhands.app_server`` (and its dependency
graph) when unit tests only need to exercise the optional-import helpers.
"""

import importlib


def _import_optional_agenthub() -> None:
    """Load optional agenthub package if available.

    `openhands.agenthub` is not always installed in local dev/test environments.
    Keep callers importable when that exact optional package is missing, but re-raise
    all other import errors to avoid hiding real failures.
    """
    try:
        importlib.import_module("openhands.agenthub")
    except ModuleNotFoundError as exc:
        if exc.name != "openhands.agenthub":
            raise


def _load_optional_security_api_router():
    """Load legacy security catch-all router when ``routes.security`` is present.

    Keeps ``openhands.server.app`` importable if that module is missing (e.g. partial
    tree or a bad merge) while still surfacing unrelated import errors.
    """
    try:
        mod = importlib.import_module("openhands.server.routes.security")
    except ModuleNotFoundError as exc:
        if exc.name != "openhands.server.routes.security":
            raise
        return None
    return mod.app
