"""Regression tests for ``openhands.app_server.services.db_session``.

These tests lock in a specific contract: calling ``depends_db_session()`` at
module load time must NOT trigger ``get_global_config()``. If it does, an
``OH_LLM_MODEL_KIND`` value that dynamically imports a feature module whose
body uses ``depends_db_session()`` as a default argument re-enters the
in-flight ``get_global_config()`` and recurses until the stack blows --
exactly the crashloop seen in OpenHands/deploy#4315.
"""

import sys
import types

from fastapi.params import Depends

from openhands.app_server.services.db_session import (
    _yield_db_session,
    depends_db_session,
)


def test_depends_db_session_returns_depends():
    """``depends_db_session()`` returns a FastAPI ``Depends`` instance."""
    result = depends_db_session()
    assert isinstance(result, Depends)
    # Must wrap the stable module-level callable, not a freshly bound method
    # closed over a (possibly half-initialised) config singleton.
    assert result.dependency is _yield_db_session


def test_depends_db_session_does_not_resolve_config_at_construction():
    """Constructing the ``Depends`` must NOT trigger the lazy ``config`` import.

    This is the bug that caused the enterprise-server pod to crashloop with
    ``OH_LLM_MODEL_KIND=server.verified_models.verified_model_router.SaaSLLMModelServiceInjector``:
    ``verified_model_service`` is dynamically imported by ``env_parser`` while
    ``config_from_env`` is still running, evaluates ``depends_db_session()``
    as a default argument, and -- in the buggy version -- called
    ``get_global_config()`` while ``_global_config`` was still ``None``,
    re-entering the same initialiser and recursing until ``RecursionError``.

    We install a sentinel ``openhands.app_server.config`` module whose
    attribute access blows up; ``depends_db_session()`` must succeed without
    ever touching it. (``_yield_db_session`` will only touch it per request.)
    """
    original = sys.modules.get('openhands.app_server.config')
    sentinel = types.ModuleType('openhands.app_server.config')

    def _explode():
        raise AssertionError(
            'depends_db_session() must NOT call get_global_config() '
            'at Depends-construction time'
        )

    sentinel.get_global_config = _explode  # type: ignore[attr-defined]
    sys.modules['openhands.app_server.config'] = sentinel
    try:
        # Calling depends_db_session() repeatedly must not raise.
        for _ in range(3):
            result = depends_db_session()
            assert isinstance(result, Depends)
    finally:
        if original is None:
            sys.modules.pop('openhands.app_server.config', None)
        else:
            sys.modules['openhands.app_server.config'] = original
