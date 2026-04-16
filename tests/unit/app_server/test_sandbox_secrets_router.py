"""Unit + integration tests for the sandbox settings endpoints and /users/me expose_secrets.

Tests:
- GET /api/v1/users/me?expose_secrets=true
- GET /api/v1/sandboxes/{sandbox_id}/settings/secrets
- GET /api/v1/sandboxes/{sandbox_id}/settings/secrets/{secret_name}
- Shared session_auth.validate_session_key()
- Integration tests exercising the real auth validation stack via HTTP
"""

import contextlib
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.sandbox.sandbox_models import (
    SandboxInfo,
    SandboxStatus,
    SecretNamesResponse,
)
from openhands.app_server.sandbox.sandbox_router import (
    get_secret_value,
    list_secret_names,
)
from openhands.app_server.sandbox.session_auth import (
    validate_session_key,
    validate_session_key_ownership,
)
from openhands.app_server.user.auth_user_context import AuthUserContext
from openhands.app_server.user.user_models import UserInfo
from openhands.app_server.user.user_router import get_current_user
from openhands.integrations.provider import ProviderHandler, ProviderToken
from openhands.integrations.service_types import ProviderType
from openhands.sdk.llm import LLM
from openhands.sdk.secret import StaticSecret
from openhands.sdk.settings import AgentSettings

SANDBOX_ID = 'sb-test-123'
USER_ID = 'test-user-id'


def _make_sandbox_info(
    sandbox_id: str = SANDBOX_ID,
    user_id: str | None = USER_ID,
) -> SandboxInfo:
    return SandboxInfo(
        id=sandbox_id,
        created_by_user_id=user_id,
        sandbox_spec_id='test-spec',
        status=SandboxStatus.RUNNING,
        session_api_key='session-key',
    )


def _patch_sandbox_service(return_sandbox: SandboxInfo | None):
    """Patch ``get_sandbox_service`` in ``session_auth`` to return a mock service."""
    mock_sandbox_service = AsyncMock()
    mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
        return_value=return_sandbox
    )
    ctx = patch(
        'openhands.app_server.sandbox.session_auth.get_sandbox_service',
    )
    return ctx, mock_sandbox_service


def _create_sandbox_service_context_manager(sandbox_service):
    """Create an async context manager that yields the given sandbox service."""

    @contextlib.asynccontextmanager
    async def _context_manager(state, request=None):
        yield sandbox_service

    return _context_manager


# ---------------------------------------------------------------------------
# validate_session_key (shared utility)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestValidateSessionKey:
    """Tests for the shared session_auth.validate_session_key utility."""

    async def test_rejects_missing_key(self):
        """Missing session key raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await validate_session_key(None)
        assert exc_info.value.status_code == 401
        assert 'X-Session-API-Key' in exc_info.value.detail

    async def test_rejects_empty_string_key(self):
        """Empty string session key raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await validate_session_key('')
        assert exc_info.value.status_code == 401

    async def test_rejects_invalid_key(self):
        """Session key that maps to no sandbox raises 401."""
        ctx, mock_svc = _patch_sandbox_service(None)
        with ctx as mock_get:
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_svc)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(HTTPException) as exc_info:
                await validate_session_key('bogus-key')
        assert exc_info.value.status_code == 401
        assert 'Invalid session API key' in exc_info.value.detail

    async def test_accepts_valid_key(self):
        """Valid session key returns sandbox info."""
        sandbox = _make_sandbox_info()
        ctx, mock_svc = _patch_sandbox_service(sandbox)
        with ctx as mock_get:
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_svc)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await validate_session_key('valid-key')
        assert result.id == SANDBOX_ID

    async def test_rejects_sandbox_without_user_in_saas_mode(self):
        """In SAAS mode, sandbox without created_by_user_id raises 401."""
        sandbox = _make_sandbox_info(user_id=None)
        ctx, mock_svc = _patch_sandbox_service(sandbox)
        with (
            ctx as mock_get,
            patch(
                'openhands.app_server.sandbox.session_auth.get_global_config'
            ) as mock_cfg,
        ):
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_svc)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=False)

            from openhands.server.types import AppMode

            mock_cfg.return_value.app_mode = AppMode.SAAS

            with pytest.raises(HTTPException) as exc_info:
                await validate_session_key('valid-key')
        assert exc_info.value.status_code == 401
        assert 'no user' in exc_info.value.detail


# ---------------------------------------------------------------------------
# GET /users/me?expose_secrets=true
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetCurrentUserExposeSecrets:
    """Test suite for GET /users/me?expose_secrets=true."""

    async def test_expose_secrets_returns_raw_api_key(self):
        """With valid session key, expose_secrets=true returns unmasked llm_api_key."""
        user_info = UserInfo(
            id=USER_ID,
            agent_settings=AgentSettings(
                llm=LLM(
                    model='anthropic/claude-sonnet-4-20250514',
                    api_key=SecretStr('sk-test-key-123'),
                    base_url='https://litellm.example.com',
                ),
            ),
        )
        mock_context = AsyncMock()
        mock_context.get_user_info = AsyncMock(return_value=user_info)
        mock_context.get_user_id = AsyncMock(return_value=USER_ID)

        with patch(
            'openhands.app_server.user.user_router.validate_session_key_ownership'
        ) as mock_validate:
            mock_validate.return_value = None
            result = await get_current_user(
                user_context=mock_context,
                expose_secrets=True,
                x_session_api_key='valid-key',
            )

        import json

        body = json.loads(result.body)
        sdk_vals = body['agent_settings']
        assert sdk_vals['llm']['model'] == 'anthropic/claude-sonnet-4-20250514'
        assert sdk_vals['llm']['api_key'] == 'sk-test-key-123'
        assert sdk_vals['llm']['base_url'] == 'https://litellm.example.com'

    async def test_expose_secrets_rejects_missing_session_key(self):
        """expose_secrets=true without X-Session-API-Key is rejected."""
        mock_context = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await validate_session_key_ownership(mock_context, session_api_key=None)
        assert exc_info.value.status_code == 401
        assert 'X-Session-API-Key' in exc_info.value.detail

    async def test_expose_secrets_rejects_wrong_user(self):
        """expose_secrets=true with session key from different user is rejected."""
        mock_context = AsyncMock()
        mock_context.get_user_id = AsyncMock(return_value='user-A')

        other_user_sandbox = _make_sandbox_info(user_id='user-B')

        ctx, mock_svc = _patch_sandbox_service(other_user_sandbox)
        with ctx as mock_get, pytest.raises(HTTPException) as exc_info:
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_svc)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=False)

            await validate_session_key_ownership(
                mock_context, session_api_key='stolen-key'
            )

        assert exc_info.value.status_code == 403

    async def test_expose_secrets_rejects_unknown_caller(self):
        """If caller_id cannot be determined, reject with 401."""
        mock_context = AsyncMock()
        mock_context.get_user_id = AsyncMock(return_value=None)

        sandbox = _make_sandbox_info(user_id='user-B')

        ctx, mock_svc = _patch_sandbox_service(sandbox)
        with ctx as mock_get, pytest.raises(HTTPException) as exc_info:
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_svc)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=False)

            await validate_session_key_ownership(
                mock_context, session_api_key='some-key'
            )

        assert exc_info.value.status_code == 401
        assert 'Cannot determine authenticated user' in exc_info.value.detail

    async def test_default_masks_api_key(self):
        """Without expose_secrets, llm_api_key is masked (no session key needed)."""
        user_info = UserInfo(
            id=USER_ID,
            agent_settings=AgentSettings(
                llm=LLM(model='gpt-4o', api_key=SecretStr('sk-test-key-123')),
            ),
        )
        mock_context = AsyncMock()
        mock_context.get_user_info = AsyncMock(return_value=user_info)

        result = await get_current_user(
            user_context=mock_context, expose_secrets=False, x_session_api_key=None
        )

        # Returns UserInfo directly (FastAPI will serialize with masking)
        assert isinstance(result, UserInfo)
        dumped = result.model_dump(mode='json')
        assert dumped['agent_settings']['llm']['api_key'] != 'sk-test-key-123'
        assert dumped['agent_settings']['llm']['api_key'] == '**********'


# ---------------------------------------------------------------------------
# GET /sandboxes/{sandbox_id}/settings/secrets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestListSecretNames:
    """Test suite for GET /sandboxes/{sandbox_id}/settings/secrets."""

    async def test_returns_secret_names_without_values(self):
        """Response contains names and descriptions, NOT raw values."""
        secrets = {
            'GITHUB_TOKEN': StaticSecret(
                value=SecretStr('ghp_test123'),
                description='GitHub personal access token',
            ),
            'MY_API_KEY': StaticSecret(
                value=SecretStr('my-api-key-value'),
                description='Custom API key',
            ),
        }
        sandbox_info = _make_sandbox_info()

        with patch(
            'openhands.app_server.sandbox.sandbox_router._get_user_context'
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_secrets = AsyncMock(return_value=secrets)
            ctx.get_provider_tokens = AsyncMock(return_value={})
            mock_ctx.return_value = ctx

            result = await list_secret_names(sandbox_info=sandbox_info)

        assert isinstance(result, SecretNamesResponse)
        assert len(result.secrets) == 2
        names = {s.name for s in result.secrets}
        assert 'GITHUB_TOKEN' in names
        assert 'MY_API_KEY' in names

        gh = next(s for s in result.secrets if s.name == 'GITHUB_TOKEN')
        assert gh.description == 'GitHub personal access token'
        # Verify no 'value' field is exposed
        assert not hasattr(gh, 'value')

    async def test_returns_empty_when_no_secrets(self):
        sandbox_info = _make_sandbox_info()

        with patch(
            'openhands.app_server.sandbox.sandbox_router._get_user_context'
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_secrets = AsyncMock(return_value={})
            ctx.get_provider_tokens = AsyncMock(return_value={})
            mock_ctx.return_value = ctx

            result = await list_secret_names(sandbox_info=sandbox_info)

        assert len(result.secrets) == 0


# ---------------------------------------------------------------------------
# GET /sandboxes/{sandbox_id}/settings/secrets/{name}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetSecretValue:
    """Test suite for GET /sandboxes/{sandbox_id}/settings/secrets/{name}."""

    async def test_returns_raw_secret_value(self):
        """Raw secret value returned as plain text."""
        secrets = {
            'GITHUB_TOKEN': StaticSecret(
                value=SecretStr('ghp_actual_secret'),
                description='GitHub token',
            ),
        }
        sandbox_info = _make_sandbox_info()

        with patch(
            'openhands.app_server.sandbox.sandbox_router._get_user_context'
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_secrets = AsyncMock(return_value=secrets)
            ctx.get_provider_tokens = AsyncMock(return_value={})
            mock_ctx.return_value = ctx

            response = await get_secret_value(
                secret_name='GITHUB_TOKEN',
                sandbox_info=sandbox_info,
            )

        assert response.body == b'ghp_actual_secret'
        assert response.media_type == 'text/plain'

    async def test_returns_404_for_unknown_secret(self):
        """404 when requested secret doesn't exist in custom secrets or provider tokens."""
        sandbox_info = _make_sandbox_info()

        with patch(
            'openhands.app_server.sandbox.sandbox_router._get_user_context'
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_secrets = AsyncMock(return_value={})
            ctx.get_provider_tokens = AsyncMock(return_value={})
            mock_ctx.return_value = ctx

            with pytest.raises(HTTPException) as exc_info:
                await get_secret_value(
                    secret_name='NONEXISTENT',
                    sandbox_info=sandbox_info,
                )

        assert exc_info.value.status_code == 404

    async def test_returns_404_for_none_value_secret(self):
        """404 when secret exists but has None value."""
        secrets = {
            'EMPTY_SECRET': StaticSecret(value=None),
        }
        sandbox_info = _make_sandbox_info()

        with patch(
            'openhands.app_server.sandbox.sandbox_router._get_user_context'
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_secrets = AsyncMock(return_value=secrets)
            ctx.get_provider_tokens = AsyncMock(return_value={})
            mock_ctx.return_value = ctx

            with pytest.raises(HTTPException) as exc_info:
                await get_secret_value(
                    secret_name='EMPTY_SECRET',
                    sandbox_info=sandbox_info,
                )

        assert exc_info.value.status_code == 404


# ===========================================================================
# Integration tests — real HTTP requests through real auth validation logic.
#
# Only the data layer (sandbox service, user context) is mocked.
# The session key validation, ownership checks, and FastAPI routing are REAL.
# ===========================================================================


def _build_integration_test_app(
    mock_user_context: AsyncMock | None = None,
) -> FastAPI:
    """Build a minimal FastAPI app with the real user and sandbox routers.

    The ``depends_user_context`` dependency is overridden with a mock, but the
    session key validation logic in ``validate_session_key`` and
    ``validate_session_key_ownership`` runs unmodified.

    Router-level dependencies (e.g. ``check_session_api_key`` from ``SESSION_API_KEY``
    env var) are overridden to no-ops so we can exercise the endpoint-level auth logic
    in isolation.
    """
    from openhands.app_server.sandbox.sandbox_router import (
        router as sandbox_router,
    )
    from openhands.app_server.user.user_router import router as user_router
    from openhands.app_server.utils.dependencies import check_session_api_key

    app = FastAPI()

    # Disable router-level auth (SESSION_API_KEY check) — we're testing the
    # endpoint-level session key validation, not the router middleware.
    app.dependency_overrides[check_session_api_key] = lambda: None

    if mock_user_context is not None:
        from openhands.app_server.user.user_router import user_dependency

        app.dependency_overrides[user_dependency.dependency] = lambda: mock_user_context

    app.include_router(user_router, prefix='/api/v1')
    app.include_router(sandbox_router, prefix='/api/v1')
    return app


class TestExposeSecretsIntegration:
    """Integration tests for /users/me?expose_secrets=true via real HTTP.

    These tests exercise the full auth validation stack:
    - validate_session_key (real)
    - validate_session_key_ownership (real)
    - ownership check (real)
    Only the data layer (sandbox service lookup, user context) is mocked.
    """

    def test_expose_secrets_without_session_key_returns_401(self):
        """Bearer token alone cannot expose secrets (no X-Session-API-Key)."""
        mock_user_ctx = AsyncMock()
        mock_user_ctx.get_user_info = AsyncMock(
            return_value=UserInfo(
                id=USER_ID,
                agent_settings=AgentSettings(
                    llm=LLM(model='gpt-4o', api_key=SecretStr('sk-secret-123')),
                ),
            )
        )
        mock_user_ctx.get_user_id = AsyncMock(return_value=USER_ID)

        app = _build_integration_test_app(mock_user_ctx)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get('/api/v1/users/me', params={'expose_secrets': 'true'})

        assert response.status_code == 401
        assert 'X-Session-API-Key' in response.json()['detail']

    def test_expose_secrets_with_invalid_session_key_returns_401(self):
        """Invalid session key (no matching sandbox) is rejected."""
        mock_user_ctx = AsyncMock()
        mock_user_ctx.get_user_info = AsyncMock(
            return_value=UserInfo(
                id=USER_ID,
                agent_settings=AgentSettings(
                    llm=LLM(model='gpt-4o', api_key=SecretStr('sk-secret-123')),
                ),
            )
        )
        mock_user_ctx.get_user_id = AsyncMock(return_value=USER_ID)

        mock_sandbox_svc = AsyncMock()
        mock_sandbox_svc.get_sandbox_by_session_api_key = AsyncMock(return_value=None)

        app = _build_integration_test_app(mock_user_ctx)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            'openhands.app_server.sandbox.session_auth.get_sandbox_service',
            _create_sandbox_service_context_manager(mock_sandbox_svc),
        ):
            response = client.get(
                '/api/v1/users/me',
                params={'expose_secrets': 'true'},
                headers={'X-Session-API-Key': 'bogus-key'},
            )

        assert response.status_code == 401
        assert 'Invalid session API key' in response.json()['detail']

    def test_expose_secrets_with_wrong_user_returns_403(self):
        """Session key from a different user's sandbox is rejected."""
        mock_user_ctx = AsyncMock()
        mock_user_ctx.get_user_info = AsyncMock(
            return_value=UserInfo(
                id='user-A',
                agent_settings=AgentSettings(
                    llm=LLM(model='gpt-4o', api_key=SecretStr('sk-secret-123')),
                ),
            )
        )
        mock_user_ctx.get_user_id = AsyncMock(return_value='user-A')

        # Sandbox owned by user-B
        sandbox_b = _make_sandbox_info(user_id='user-B')
        mock_sandbox_svc = AsyncMock()
        mock_sandbox_svc.get_sandbox_by_session_api_key = AsyncMock(
            return_value=sandbox_b
        )

        app = _build_integration_test_app(mock_user_ctx)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            'openhands.app_server.sandbox.session_auth.get_sandbox_service',
            _create_sandbox_service_context_manager(mock_sandbox_svc),
        ):
            response = client.get(
                '/api/v1/users/me',
                params={'expose_secrets': 'true'},
                headers={'X-Session-API-Key': 'stolen-key'},
            )

        assert response.status_code == 403
        assert 'does not belong' in response.json()['detail']

    def test_expose_secrets_valid_dual_auth_returns_200_unmasked(self):
        """Valid Bearer + valid session key owned by same user → 200 with secrets."""
        mock_user_ctx = AsyncMock()
        mock_user_ctx.get_user_info = AsyncMock(
            return_value=UserInfo(
                id=USER_ID,
                agent_settings=AgentSettings(
                    llm=LLM(
                        model='anthropic/claude-sonnet-4-20250514',
                        api_key=SecretStr('sk-real-secret'),
                        base_url='https://litellm.example.com',
                    ),
                ),
            )
        )
        mock_user_ctx.get_user_id = AsyncMock(return_value=USER_ID)

        sandbox = _make_sandbox_info(user_id=USER_ID)
        mock_sandbox_svc = AsyncMock()
        mock_sandbox_svc.get_sandbox_by_session_api_key = AsyncMock(
            return_value=sandbox
        )

        app = _build_integration_test_app(mock_user_ctx)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            'openhands.app_server.sandbox.session_auth.get_sandbox_service',
            _create_sandbox_service_context_manager(mock_sandbox_svc),
        ):
            response = client.get(
                '/api/v1/users/me',
                params={'expose_secrets': 'true'},
                headers={'X-Session-API-Key': 'valid-key'},
            )

        assert response.status_code == 200
        user = UserInfo.model_validate_json(response.text)
        assert user.agent_settings.llm.api_key.get_secret_value() == 'sk-real-secret'
        assert user.agent_settings.llm.model == 'anthropic/claude-sonnet-4-20250514'
        assert user.agent_settings.llm.base_url == 'https://litellm.example.com'

    def test_default_masks_secrets_via_http(self):
        """Without expose_secrets, secrets are in agent_settings."""
        mock_user_ctx = AsyncMock()
        mock_user_ctx.get_user_info = AsyncMock(
            return_value=UserInfo(
                id=USER_ID,
                agent_settings=AgentSettings(
                    llm=LLM(
                        model='gpt-4o',
                        api_key=SecretStr('sk-should-be-masked'),
                    ),
                ),
            )
        )

        app = _build_integration_test_app(mock_user_ctx)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get('/api/v1/users/me')

        assert response.status_code == 200
        user = UserInfo.model_validate_json(response.text)
        # Masked secrets are stripped to None on deserialization
        assert user.agent_settings.llm.api_key is None


class TestSandboxSecretsIntegration:
    """Integration tests for sandbox-scoped secrets endpoints via real HTTP.

    The session key validation in ``_valid_sandbox_from_session_key`` runs
    unmodified — only the sandbox service (database) is mocked.
    """

    def test_secrets_list_without_session_key_returns_401(self):
        """Missing X-Session-API-Key on secrets endpoint is rejected."""
        app = _build_integration_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get(f'/api/v1/sandboxes/{SANDBOX_ID}/settings/secrets')

        assert response.status_code == 401
        assert 'X-Session-API-Key' in response.json()['detail']

    def test_secrets_list_with_invalid_session_key_returns_401(self):
        """Invalid session key on secrets endpoint is rejected."""
        app = _build_integration_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        mock_sandbox_svc = AsyncMock()
        mock_sandbox_svc.get_sandbox_by_session_api_key = AsyncMock(return_value=None)

        with patch(
            'openhands.app_server.sandbox.session_auth.get_sandbox_service',
            _create_sandbox_service_context_manager(mock_sandbox_svc),
        ):
            response = client.get(
                f'/api/v1/sandboxes/{SANDBOX_ID}/settings/secrets',
                headers={'X-Session-API-Key': 'bogus'},
            )

        assert response.status_code == 401
        assert 'Invalid session API key' in response.json()['detail']

    def test_secrets_list_with_mismatched_sandbox_id_returns_403(self):
        """Session key maps to a different sandbox than the URL path → 403."""
        app = _build_integration_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Session key maps to sandbox "other-sandbox", but URL says SANDBOX_ID
        other_sandbox = _make_sandbox_info(sandbox_id='other-sandbox')
        mock_sandbox_svc = AsyncMock()
        mock_sandbox_svc.get_sandbox_by_session_api_key = AsyncMock(
            return_value=other_sandbox
        )

        with patch(
            'openhands.app_server.sandbox.session_auth.get_sandbox_service',
            _create_sandbox_service_context_manager(mock_sandbox_svc),
        ):
            response = client.get(
                f'/api/v1/sandboxes/{SANDBOX_ID}/settings/secrets',
                headers={'X-Session-API-Key': 'valid-key'},
            )

        assert response.status_code == 403
        assert 'does not match' in response.json()['detail']

    def test_sandbox_without_user_returns_401_for_secret_value(self):
        """Sandbox with no owning user → 401 when fetching a secret value."""
        app = _build_integration_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Sandbox exists but has no owning user
        sandbox_no_user = _make_sandbox_info(user_id=None)
        mock_sandbox_svc = AsyncMock()
        mock_sandbox_svc.get_sandbox_by_session_api_key = AsyncMock(
            return_value=sandbox_no_user
        )

        with patch(
            'openhands.app_server.sandbox.session_auth.get_sandbox_service',
            _create_sandbox_service_context_manager(mock_sandbox_svc),
        ):
            response = client.get(
                f'/api/v1/sandboxes/{SANDBOX_ID}/settings/secrets/MY_SECRET',
                headers={'X-Session-API-Key': 'valid-key'},
            )

        # _get_user_context raises 401 because created_by_user_id is None
        assert response.status_code == 401
        assert 'no associated user' in response.json()['detail']


# ---------------------------------------------------------------------------
# Provider tokens in sandbox secrets endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProviderTokensInEndpoints:
    """Verify that sandbox secrets endpoints include provider tokens resolved lazily."""

    async def test_get_provider_tokens_as_env_vars(self):
        """get_provider_tokens(as_env_vars=True) returns fresh values keyed by env name."""
        mock_user_auth = AsyncMock()
        mock_user_auth.get_provider_tokens = AsyncMock(
            return_value={
                ProviderType.GITHUB: ProviderToken(token=SecretStr('ghp_test123')),
                ProviderType.GITLAB: ProviderToken(token=SecretStr('glpat-test456')),
            }
        )

        ctx = AuthUserContext(user_auth=mock_user_auth)
        result = await ctx.get_provider_tokens(as_env_vars=True)

        gh_key = ProviderHandler.get_provider_env_key(ProviderType.GITHUB)
        gl_key = ProviderHandler.get_provider_env_key(ProviderType.GITLAB)
        assert result[gh_key] == 'ghp_test123'
        assert result[gl_key] == 'glpat-test456'

    async def test_empty_provider_tokens_excluded(self):
        """Provider tokens with empty token values are excluded."""
        mock_user_auth = AsyncMock()
        mock_user_auth.get_provider_tokens = AsyncMock(
            return_value={
                ProviderType.GITHUB: ProviderToken(token=SecretStr('')),
            }
        )

        ctx = AuthUserContext(user_auth=mock_user_auth)
        result = await ctx.get_provider_tokens(as_env_vars=True)

        gh_key = ProviderHandler.get_provider_env_key(ProviderType.GITHUB)
        assert gh_key not in result

    async def test_none_provider_tokens_returns_empty(self):
        """get_provider_tokens(as_env_vars=True) with None tokens yields empty dict."""
        mock_user_auth = AsyncMock()
        mock_user_auth.get_provider_tokens = AsyncMock(return_value=None)

        ctx = AuthUserContext(user_auth=mock_user_auth)
        result = await ctx.get_provider_tokens(as_env_vars=True)
        assert result == {}

    async def test_list_secret_names_includes_provider_tokens(self):
        """list_secret_names returns both custom secrets and provider token names."""
        custom_secrets = {
            'MY_KEY': StaticSecret(
                value=SecretStr('my-value'), description='custom key'
            ),
        }
        gh_key = ProviderHandler.get_provider_env_key(ProviderType.GITHUB)
        provider_env_vars = {gh_key: 'ghp_test123'}

        sandbox_info = _make_sandbox_info()

        with patch(
            'openhands.app_server.sandbox.sandbox_router._get_user_context'
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_secrets = AsyncMock(return_value=custom_secrets)
            ctx.get_provider_tokens = AsyncMock(return_value=provider_env_vars)
            mock_ctx.return_value = ctx

            result = await list_secret_names(sandbox_info=sandbox_info)

        names = {s.name for s in result.secrets}
        assert 'MY_KEY' in names
        assert gh_key in names
        assert len(result.secrets) == 2

    async def test_get_secret_value_resolves_provider_token(self):
        """get_secret_value falls back to provider tokens when not in custom secrets."""
        gh_key = ProviderHandler.get_provider_env_key(ProviderType.GITHUB)
        sandbox_info = _make_sandbox_info()

        with patch(
            'openhands.app_server.sandbox.sandbox_router._get_user_context'
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_secrets = AsyncMock(return_value={})
            ctx.get_provider_tokens = AsyncMock(
                return_value={gh_key: 'ghp_fresh_token'}
            )
            mock_ctx.return_value = ctx

            response = await get_secret_value(
                secret_name=gh_key, sandbox_info=sandbox_info
            )

        assert response.body == b'ghp_fresh_token'
        assert response.media_type == 'text/plain'

    async def test_custom_secret_takes_priority_over_provider_token(self):
        """If a custom secret has the same name, it takes priority."""
        gh_key = ProviderHandler.get_provider_env_key(ProviderType.GITHUB)
        sandbox_info = _make_sandbox_info()

        with patch(
            'openhands.app_server.sandbox.sandbox_router._get_user_context'
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_secrets = AsyncMock(
                return_value={
                    gh_key: StaticSecret(
                        value=SecretStr('custom-override'),
                        description='user override',
                    )
                }
            )
            # Provider token should NOT be called since custom secret matches
            ctx.get_provider_tokens = AsyncMock(return_value={gh_key: 'provider-value'})
            mock_ctx.return_value = ctx

            response = await get_secret_value(
                secret_name=gh_key, sandbox_info=sandbox_info
            )

        assert response.body == b'custom-override'
