"""Databricks AI Gateway model picker — tier-2 dynamic discovery endpoint.

Exposes the merged curated + discovered model list from the openhands-sdk
Databricks provider (``get_picker_entries``). Used by the OpenHands web UI
settings screen to populate the Databricks model dropdown.

Endpoint:
  GET /api/v1/databricks/models

Behavior (two-tier, matches ``get_picker_entries``):
  1. **Curated tier (always)** — the hand-picked Claude / GPT / Gemini set from
     ``CURATED_DATABRICKS_MODELS``. Returned even if the caller has no
     Databricks credentials, so the picker is never empty.
  2. **Discovered tier (best-effort)** — if we can resolve workspace host +
     auth from the authenticated user's context (stored PAT, session U2M
     token, or env fallback), we call
     ``list_chat_endpoints`` and merge the live endpoints (FOUNDATION_MODEL_API
     + EXTERNAL_MODEL) on top of the curated set. Any error here is swallowed;
     the curated list is always returned so the UI stays usable offline /
     during outages.

Security:
  * Requires the standard OpenHands auth dependency (same as ``/users/me``).
  * Tokens are **never** accepted via query param (would leak into access logs
    and browser history); they are only read from the authenticated server-side
    session / user settings.
  * The response contains only model ids + metadata — no secrets, no tokens.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Query, Request

from openhands.app_server.auth.databricks_routes import _get_u2m_tokens
from openhands.app_server.config import depends_user_context
from openhands.app_server.user.user_context import UserContext
from openhands.server.dependencies import get_dependencies

_logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/databricks',
    tags=['databricks-models'],
    dependencies=get_dependencies(),
)

user_dependency = depends_user_context()


def _resolve_host(
    explicit_host: str | None,
    user_base_url: str | None,
) -> str | None:
    """Pick a workspace host to query against. Query param > user settings > env.

    Returns ``None`` if nothing sensible is configured. The discovery tier is
    then skipped (curated-only response).
    """
    for candidate in (explicit_host, user_base_url, os.environ.get('DATABRICKS_HOST')):
        if candidate:
            host = candidate.strip().rstrip('/')
            if host.startswith('http://') or host.startswith('https://'):
                return host
    return None


def _resolve_ai_gateway_host(
    user_ai_gateway_host: str | None,
) -> str | None:
    """Pick a dedicated AI Gateway host when one is explicitly configured.

    Priority: user setting > DATABRICKS_AI_GATEWAY_HOST env var.
    Returns ``None`` when no dedicated gateway URL is set (workspace host is used
    as the gateway base by default in the SDK).
    """
    for candidate in (
        user_ai_gateway_host,
        os.environ.get('DATABRICKS_AI_GATEWAY_HOST'),
    ):
        if candidate:
            host = candidate.strip().rstrip('/')
            if host.startswith('http://') or host.startswith('https://'):
                return host
    return None


def _resolve_token(
    request: Request,
    user_api_key: Any,
) -> str | None:
    """Best-effort token resolution from the authenticated context.

    Priority:
      1. Session-stored U2M access token (set by ``/auth/databricks/callback``).
      2. User's stored ``llm_api_key`` (PAT).
      3. ``DATABRICKS_TOKEN`` / ``DATABRICKS_ACCESS_TOKEN`` env fallback
         (useful for local dev, service-account deployments).

    We never accept tokens via query params.
    """
    # 1. U2M session token (preferred — scoped to the user's browser session).
    # Tokens live in the server-side store; only a session ID is in the cookie.
    u2m = _get_u2m_tokens(request)
    if isinstance(u2m, dict):
        access = u2m.get('access_token')
        if isinstance(access, str) and access:
            return access

    # 2. User-stored PAT.
    if user_api_key is not None:
        try:
            pat = user_api_key.get_secret_value()  # SecretStr
        except AttributeError:
            pat = str(user_api_key) if user_api_key else None
        if pat:
            return pat

    # 3. Env fallback.
    env_tok = os.environ.get('DATABRICKS_TOKEN') or os.environ.get(
        'DATABRICKS_ACCESS_TOKEN'
    )
    return env_tok or None


def _serialize_entry(entry: Any) -> dict[str, Any]:
    """Serialize a ``ModelPickerEntry`` for JSON transport.

    We build the dict by hand (rather than ``dataclasses.asdict``) so the
    ``ProviderFamily`` enum is emitted as its string value, and so we pin the
    JSON schema independently of the SDK dataclass shape.
    """
    return {
        'qualified_name': entry.qualified_name,
        'name': entry.name,
        'family': entry.family.value,
        'source': entry.source,
        'endpoint_type': entry.endpoint_type,
        'ready': entry.ready,
        'recommended': entry.recommended,
    }


@router.get('/models')
async def list_databricks_models(
    request: Request,
    host: str | None = Query(
        default=None,
        description=(
            'Databricks workspace URL (https://…). Optional — falls back to the '
            "user's stored llm_base_url, then DATABRICKS_HOST env."
        ),
    ),
    include_discovered: bool = Query(
        default=True,
        description=(
            'Merge live endpoints from the workspace on top of the curated list. '
            'Set to false for an offline picker that only shows the curated tier.'
        ),
    ),
    user_context: UserContext = user_dependency,
) -> dict[str, Any]:
    """Return the merged curated + discovered Databricks picker list.

    The curated tier is always present. The discovered tier is added best-effort
    — any auth/network failure logs and degrades to curated-only. Never raises.
    """
    # Import inside the handler so the route stays importable in environments
    # where the Databricks extra isn't installed. If the import fails, we
    # return a static empty-curated response rather than a 500.
    try:
        from openhands.sdk.llm.providers.databricks import (
            AuthStrategy,
            DatabricksCredentials,
            get_picker_entries,
        )
    except ImportError as exc:
        _logger.warning(
            'databricks_models_route_sdk_missing', extra={'error': str(exc)}
        )
        return {
            'entries': [],
            'source': 'unavailable',
            'reason': 'openhands-sdk Databricks provider not installed',
        }

    # Settings may not exist yet (first-time setup before any settings are saved).
    # Degrade gracefully to env-var-only auth rather than returning a 500.
    try:
        user = await user_context.get_user_info()
    except Exception:
        user = None
    user_base_url = getattr(user, 'llm_base_url', None) if user else None
    user_api_key = getattr(user, 'llm_api_key', None) if user else None

    resolved_host = _resolve_host(host, user_base_url)
    credentials = None
    if include_discovered and resolved_host:
        token = _resolve_token(request, user_api_key)
        if token:
            try:
                credentials = DatabricksCredentials(
                    host=resolved_host,
                    get_token=lambda t=token: t,
                    auth_method=AuthStrategy.PAT,
                )
            except Exception as exc:  # defensive: never 500 the picker
                _logger.warning(
                    'databricks_models_route_cred_failure',
                    extra={'error': str(exc)},
                )
                credentials = None

    entries = get_picker_entries(
        credentials=credentials,
        include_curated=True,
        include_discovered=include_discovered,
    )

    # Report "curated+discovered" only when the live workspace probe actually
    # succeeded and returned at least one "discovered" entry.
    # Previously we checked `credentials is not None`, which is wrong: the
    # backend holds a valid session token, so credentials exist even when the
    # workspace hostname is wrong or unreachable — get_picker_entries silently
    # swallows the error and returns only curated entries in that case.
    has_discovered = any('discovered' in (e.source or '') for e in entries)
    tier = 'curated+discovered' if has_discovered else 'curated'

    return {
        'entries': [_serialize_entry(e) for e in entries],
        'source': tier,
        'host': resolved_host,
    }
