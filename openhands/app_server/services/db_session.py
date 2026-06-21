"""FastAPI dependency-injection helpers for the global ``AsyncSession``.

These helpers live here (next to ``db_session_injector``) rather than in
``openhands.app_server.config`` so that feature-layer modules that need an
``AsyncSession`` do not have to import ``app_server.config``.

That separation is important because ``app_server.config`` imports
``openhands.agent_server.env_parser``, and ``env_parser`` is what
dynamically loads classes named by ``OH_LLM_MODEL_KIND`` (and similar
discriminated-union env vars) at runtime. Some of those classes live in
feature packages such as ``server.verified_models``. If a feature module
also imports from ``app_server.config`` at module top level, the import
graph closes into a cycle:

    feature module
        -> openhands.app_server.config
        -> openhands.agent_server.env_parser
        -> (dynamic) feature module

The cycle is latent today because the dynamic edge is only traversed
lazily, but it would surface as an ``ImportError`` the moment anything
caused the feature module to be imported before ``app_server.config``
finished initialising. Keeping these DI helpers in a dependency-free
module lets feature code obtain an ``AsyncSession`` without putting
``app_server.config`` on its static import path.

``get_global_config`` is imported lazily inside each function body for
the same reason: importing this module must not pull in
``app_server.config``.
"""

from typing import AsyncContextManager, AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.app_server.services.injector import InjectorState


def get_db_session(
    state: InjectorState, request: Request | None = None
) -> AsyncContextManager[AsyncSession]:
    """Return an async context manager yielding the request-scoped ``AsyncSession``."""
    from openhands.app_server.config import get_global_config

    return get_global_config().db_session.context(state, request)


async def _yield_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Per-request FastAPI dependency yielding the request-scoped ``AsyncSession``.

    ``get_global_config`` is looked up at request time -- not at module load --
    so that this module is safe to evaluate from inside ``config_from_env``.
    Concretely, when ``OH_LLM_MODEL_KIND`` (or a similar discriminated-union env
    var) names a feature class whose module body calls ``depends_db_session()``
    as a default argument, ``env_parser`` will dynamically import that module
    while we are still constructing the global config. Calling
    ``get_global_config()`` eagerly here would re-enter the unfinished
    initialiser and recurse until the stack blows.
    """
    from openhands.app_server.config import get_global_config

    async for db_session in get_global_config().db_session.depends(request):
        yield db_session


def depends_db_session():
    """FastAPI ``Depends(...)`` factory for the request-scoped ``AsyncSession``.

    Returns a stable ``Depends`` object that defers the ``get_global_config()``
    lookup to request time. Safe to call from module top level (e.g. as a
    default argument), even while ``config_from_env`` is still running.
    """
    return Depends(_yield_db_session)
