"""Unit tests for API keys routes, focusing on BYOR key validation and retrieval."""

import contextlib
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from server.auth.saas_user_auth import SaasUserAuth
from server.constants import BYOR_KEY_ALIAS_PATTERN, ORG_SETTINGS_VERSION
from server.routes.api_keys import (
    ByorPermittedResponse,
    CurrentApiKeyResponse,
    LlmApiKeyResponse,
    ManagedLlmApiKeyRefreshResponse,
    _create_byor_key_alias,
    check_byor_permitted,
    delete_byor_key_from_litellm,
    generate_byor_key,
    get_current_api_key,
    get_llm_api_key_for_byor,
    refresh_managed_llm_api_key,
)
from storage.lite_llm_manager import LiteLlmManager, get_openhands_cloud_key_alias
from storage.org import Org
from storage.org_member import OrgMember
from storage.role import Role
from storage.saas_settings_store import (
    ManagedLlmKeyConfig,
    ManagedLlmKeyStatus,
    SaasSettingsStore,
    managed_llm_key_config_from_model,
)
from storage.user import User

from openhands.app_server.user_auth.user_auth import AuthType


class TestVerifyByorKeyInLitellm:
    """Test the verify_byor_key_in_litellm function."""

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_valid_key_returns_true(self, mock_client_class):
        """Test that a valid key (200 response) returns True."""
        # Arrange
        byor_key = 'sk-valid-key-123'
        user_id = 'user-123'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is True
        mock_client.get.assert_called_once_with(
            'https://litellm.example.com/v1/models',
            headers={'Authorization': f'Bearer {byor_key}'},
        )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_invalid_key_401_returns_false(self, mock_client_class):
        """Test that an invalid key (401 response) returns False."""
        # Arrange
        byor_key = 'sk-invalid-key-123'
        user_id = 'user-123'
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_invalid_key_403_returns_false(self, mock_client_class):
        """Test that an invalid key (403 response) returns False."""
        # Arrange
        byor_key = 'sk-forbidden-key-123'
        user_id = 'user-123'
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_server_error_returns_false(self, mock_client_class):
        """Test that a server error (500) returns False to ensure key validity."""
        # Arrange
        byor_key = 'sk-key-123'
        user_id = 'user-123'
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.is_success = False
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_timeout_returns_false(self, mock_client_class):
        """Test that a timeout returns False to ensure key validity."""
        # Arrange
        byor_key = 'sk-key-123'
        user_id = 'user-123'
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.side_effect = httpx.TimeoutException('Request timed out')
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    @patch('storage.lite_llm_manager.httpx.AsyncClient')
    async def test_verify_network_error_returns_false(self, mock_client_class):
        """Test that a network error returns False to ensure key validity."""
        # Arrange
        byor_key = 'sk-key-123'
        user_id = 'user-123'
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.side_effect = httpx.NetworkError('Network error')
        mock_client_class.return_value = mock_client

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', None)
    async def test_verify_missing_api_url_returns_false(self):
        """Test that missing LITE_LLM_API_URL returns False."""
        # Arrange
        byor_key = 'sk-key-123'
        user_id = 'user-123'

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LITE_LLM_API_URL', 'https://litellm.example.com')
    async def test_verify_empty_key_returns_false(self):
        """Test that empty key returns False."""
        # Arrange
        byor_key = ''
        user_id = 'user-123'

        # Act
        result = await LiteLlmManager.verify_key(byor_key, user_id)

        # Assert
        assert result is False


class TestGetLlmApiKeyForByor:
    """Test the get_llm_api_key_for_byor endpoint."""

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.store_byor_key_in_db')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_no_key_in_database_generates_new(
        self, mock_get_key, mock_generate_key, mock_store_key, mock_check_enabled
    ):
        """Test that when no key exists in database, a new one is generated."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        new_key = 'sk-new-generated-key'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = None
        mock_generate_key.return_value = new_key
        mock_store_key.return_value = None

        # Act
        result = await get_llm_api_key_for_byor(
            user_id=user_id, effective_org_id=org_id
        )

        # Assert
        assert result == LlmApiKeyResponse(key=new_key)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)
        mock_get_key.assert_called_once_with(user_id, org_id)
        mock_generate_key.assert_called_once_with(user_id, org_id)
        mock_store_key.assert_called_once_with(user_id, org_id, new_key)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('storage.lite_llm_manager.LiteLlmManager.verify_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_valid_key_in_database_returns_key(
        self, mock_get_key, mock_verify_key, mock_check_enabled
    ):
        """Test that when a valid key exists in database, it is returned."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        existing_key = 'sk-existing-valid-key'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = existing_key
        mock_verify_key.return_value = True

        # Act
        result = await get_llm_api_key_for_byor(
            user_id=user_id, effective_org_id=org_id
        )

        # Assert
        assert result == LlmApiKeyResponse(key=existing_key)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)
        mock_get_key.assert_called_once_with(user_id, org_id)
        mock_verify_key.assert_called_once_with(existing_key, user_id)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.store_byor_key_in_db')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.delete_byor_key_from_litellm')
    @patch('storage.lite_llm_manager.LiteLlmManager.verify_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_invalid_key_in_database_regenerates(
        self,
        mock_get_key,
        mock_verify_key,
        mock_delete_key,
        mock_generate_key,
        mock_store_key,
        mock_check_enabled,
    ):
        """Test that when an invalid key exists in database, it is regenerated."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        invalid_key = 'sk-invalid-key'
        new_key = 'sk-new-generated-key'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = invalid_key
        mock_verify_key.return_value = False
        mock_delete_key.return_value = True
        mock_generate_key.return_value = new_key
        mock_store_key.return_value = None

        # Act
        result = await get_llm_api_key_for_byor(
            user_id=user_id, effective_org_id=org_id
        )

        # Assert
        assert result == LlmApiKeyResponse(key=new_key)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)
        mock_get_key.assert_called_once_with(user_id, org_id)
        mock_verify_key.assert_called_once_with(invalid_key, user_id)
        mock_delete_key.assert_called_once_with(user_id, org_id, invalid_key)
        mock_generate_key.assert_called_once_with(user_id, org_id)
        mock_store_key.assert_called_once_with(user_id, org_id, new_key)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.store_byor_key_in_db')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.delete_byor_key_from_litellm')
    @patch('storage.lite_llm_manager.LiteLlmManager.verify_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_invalid_key_deletion_failure_still_regenerates(
        self,
        mock_get_key,
        mock_verify_key,
        mock_delete_key,
        mock_generate_key,
        mock_store_key,
        mock_check_enabled,
    ):
        """Test that even if deletion fails, regeneration still proceeds."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        invalid_key = 'sk-invalid-key'
        new_key = 'sk-new-generated-key'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = invalid_key
        mock_verify_key.return_value = False
        mock_delete_key.return_value = False  # Deletion fails
        mock_generate_key.return_value = new_key
        mock_store_key.return_value = None

        # Act
        result = await get_llm_api_key_for_byor(
            user_id=user_id, effective_org_id=org_id
        )

        # Assert
        assert result == LlmApiKeyResponse(key=new_key)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)
        mock_delete_key.assert_called_once_with(user_id, org_id, invalid_key)
        mock_generate_key.assert_called_once_with(user_id, org_id)
        mock_store_key.assert_called_once_with(user_id, org_id, new_key)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.generate_byor_key')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_key_generation_failure_raises_exception(
        self, mock_get_key, mock_generate_key, mock_check_enabled
    ):
        """Test that when key generation fails, an HTTPException is raised."""
        # Arrange
        user_id = 'user-123'
        mock_check_enabled.return_value = True
        mock_get_key.return_value = None
        mock_generate_key.return_value = None

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_llm_api_key_for_byor(user_id=user_id)

        assert exc_info.value.status_code == 500
        assert 'Failed to generate new BYOR LLM API key' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    @patch('server.routes.api_keys.get_byor_key_from_db')
    async def test_database_error_raises_exception(
        self, mock_get_key, mock_check_enabled
    ):
        """Test that database errors are properly handled."""
        # Arrange
        user_id = 'user-123'
        mock_check_enabled.return_value = True
        mock_get_key.side_effect = Exception('Database connection error')

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_llm_api_key_for_byor(user_id=user_id)

        assert exc_info.value.status_code == 500
        assert 'Failed to retrieve BYOR LLM API key' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    async def test_byor_export_disabled_returns_402(self, mock_check_enabled):
        """Test that when BYOR export is disabled, 402 is returned."""
        # Arrange
        user_id = 'user-123'
        mock_check_enabled.return_value = False

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_llm_api_key_for_byor(user_id=user_id)

        assert exc_info.value.status_code == 402
        assert 'BYOR key export is not enabled' in exc_info.value.detail


class TestRefreshManagedLlmApiKey:
    """Test the managed LLM API key refresh endpoint.

    These tests exercise the REAL managed-key lifecycle
    (``SaasSettingsStore.rotate_managed_llm_key``) against an in-memory SQLite
    database, mocking only the external LiteLLM HTTP calls. They prove the
    actual managed-config classification (from effective org+member settings,
    with org-default precedence), the OpenHands metadata attachment, and the
    persist/missing-member behavior — not stubs of the route helpers.
    """

    MANAGED_BASE_URL = 'https://litellm.example.com'

    @pytest.fixture
    def managed_env(self):
        """Point the managed base_url at a constant value for classification."""
        with (
            patch('server.constants.LITE_LLM_API_URL', self.MANAGED_BASE_URL),
            patch('storage.lite_llm_manager.LITE_LLM_API_URL', self.MANAGED_BASE_URL),
            patch(
                'storage.saas_settings_store.LITE_LLM_API_URL',
                self.MANAGED_BASE_URL,
            ),
        ):
            yield

    @staticmethod
    async def _seed(
        async_session_maker,
        *,
        llm_model='openhands/claude-sonnet-4',
        llm_base_url=None,
        member_custom=False,
        member_key='sk-old-managed-key',
        org_key=None,
    ):
        """Create a user/org/role/member with the given effective LLM config.

        The org's agent_settings carry the LLM model/base_url, so the effective
        config (org defaults merged with an empty member diff) is what
        ``load()`` resolves — exercising org-default precedence.
        Returns (user_id_str, org_id).
        """
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        llm_settings = {'model': llm_model}
        if llm_base_url is not None:
            llm_settings['base_url'] = llm_base_url

        async with async_session_maker() as session:
            role = Role(name='owner', rank=1)
            org_kwargs = {
                'id': org_id,
                'name': f'test-org-{org_id}',
                'org_version': ORG_SETTINGS_VERSION,
                'agent_settings': {'llm': llm_settings},
                'enable_proactive_conversation_starters': True,
                'sandbox_grouping_strategy': 'NO_GROUPING',
            }
            if org_key is not None:
                org_kwargs['llm_api_key'] = org_key
            org = Org(**org_kwargs)
            user = User(
                id=user_id,
                current_org_id=org_id,
                enable_sound_notifications=False,
                git_full_clone=False,
                sandbox_grouping_strategy='NO_GROUPING',
            )
            session.add_all([role, org, user])
            await session.flush()
            member = OrgMember(
                org_id=org_id,
                user_id=user_id,
                role_id=role.id,
                llm_api_key=member_key,
            )
            member.has_custom_llm_api_key = member_custom
            session.add(member)
            await session.commit()
        return str(user_id), org_id

    @staticmethod
    async def _member_key(async_session_maker, org_id, user_id):
        from sqlalchemy import select

        async with async_session_maker() as session:
            member = (
                (
                    await session.execute(
                        select(OrgMember).where(
                            OrgMember.org_id == org_id,
                            OrgMember.user_id == uuid.UUID(user_id),
                        )
                    )
                )
                .scalars()
                .one()
            )
            return member

    @staticmethod
    def _session_patches(async_session_maker):
        """Point every store's session maker at the test DB."""
        return (
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
            patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        )

    @staticmethod
    def _litellm_patches(*, generated_key='sk-new-managed-key'):
        """Patch only the external LiteLLM HTTP calls."""
        return (
            patch(
                'storage.lite_llm_manager.LiteLlmManager.delete_key_by_alias',
                new_callable=AsyncMock,
            ),
            patch(
                'storage.lite_llm_manager.LiteLlmManager.generate_key',
                new_callable=AsyncMock,
                return_value=generated_key,
            ),
            patch(
                'storage.lite_llm_manager.LiteLlmManager.delete_key',
                new_callable=AsyncMock,
            ),
        )

    @classmethod
    @contextlib.contextmanager
    def _patched(cls, async_session_maker, *, generated_key='sk-new-managed-key'):
        """Enter session + LiteLLM patches together, yielding the LiteLLM mocks.

        (mock_delete_alias, mock_generate, mock_delete_token)
        """
        with contextlib.ExitStack() as stack:
            for p in cls._session_patches(async_session_maker):
                stack.enter_context(p)
            mock_delete_alias, mock_generate, mock_delete_token = (
                stack.enter_context(p)
                for p in cls._litellm_patches(generated_key=generated_key)
            )
            yield mock_delete_alias, mock_generate, mock_delete_token

    @classmethod
    @contextlib.contextmanager
    def _patched_route(
        cls, async_session_maker, user_id, org_id, *, generated_key='sk-new-managed-key'
    ):
        """Like ``_patched`` but also patches ``get_instance`` to return a real
        ``SaasSettingsStore`` bound to the test DB, so the route exercises the
        real rotation end-to-end.
        """
        with contextlib.ExitStack() as stack:
            for p in cls._session_patches(async_session_maker):
                stack.enter_context(p)
            mock_delete_alias, mock_generate, mock_delete_token = (
                stack.enter_context(p)
                for p in cls._litellm_patches(generated_key=generated_key)
            )
            stack.enter_context(
                patch(
                    'storage.saas_settings_store.SaasSettingsStore.get_instance',
                    new_callable=AsyncMock,
                    return_value=SaasSettingsStore(user_id, effective_org_id=org_id),
                )
            )
            yield mock_delete_alias, mock_generate, mock_delete_token

    # --- pure classification (no DB) ---

    def test_managed_config_detects_managed_base_url(self):
        with patch(
            'storage.saas_settings_store.LITE_LLM_API_URL',
            'https://litellm.example.com/',
        ):
            assert managed_llm_key_config_from_model(
                'anthropic/claude-sonnet-4', 'https://litellm.example.com'
            ) == ManagedLlmKeyConfig(openhands_type=False)

    def test_managed_config_detects_openhands_model_without_base_url(self):
        with patch(
            'storage.saas_settings_store.LITE_LLM_API_URL',
            'https://litellm.example.com',
        ):
            assert managed_llm_key_config_from_model(
                'openhands/claude-sonnet-4', None
            ) == ManagedLlmKeyConfig(openhands_type=True)

    def test_managed_config_detects_openhands_model_with_all_hands_base_url(self):
        with patch(
            'storage.saas_settings_store.LITE_LLM_API_URL',
            'https://litellm.example.com',
        ):
            assert managed_llm_key_config_from_model(
                'openhands/claude-sonnet-4',
                'https://llm-proxy.staging.all-hands.dev/v1',
            ) == ManagedLlmKeyConfig(openhands_type=True)

    def test_managed_config_rejects_openhands_model_custom_base_url(self):
        with patch(
            'storage.saas_settings_store.LITE_LLM_API_URL',
            'https://litellm.example.com',
        ):
            assert (
                managed_llm_key_config_from_model(
                    'openhands/claude-sonnet-4', 'https://custom-llm.example.com'
                )
                is None
            )

    def test_managed_config_rejects_non_managed_provider_base_url(self):
        with patch(
            'storage.saas_settings_store.LITE_LLM_API_URL',
            'https://litellm.example.com',
        ):
            assert (
                managed_llm_key_config_from_model(
                    'anthropic/claude-sonnet-4', 'https://api.anthropic.com'
                )
                is None
            )

    # --- real rotate_managed_llm_key behavior ---

    @pytest.mark.asyncio
    async def test_rotate_openhands_model_attaches_openhands_metadata(
        self, async_session_maker, managed_env
    ):
        """An openhands/* effective config rotates with {'type': 'openhands'}."""
        user_id, org_id = await self._seed(async_session_maker)

        with self._patched(async_session_maker) as (
            mock_delete_alias,
            mock_generate,
            mock_delete_token,
        ):
            store = SaasSettingsStore(user_id, effective_org_id=org_id)
            rotation = await store.rotate_managed_llm_key()

        assert rotation.status == ManagedLlmKeyStatus.ROTATED
        assert rotation.openhands_type is True
        assert rotation.old_key == 'sk-old-managed-key'
        assert rotation.new_key == 'sk-new-managed-key'

        expected_alias = get_openhands_cloud_key_alias(user_id, str(org_id))
        # The alias is deleted before generating, so rotation never orphans.
        mock_delete_alias.assert_awaited_once_with(key_alias=expected_alias)
        mock_generate.assert_awaited_once_with(
            user_id, str(org_id), expected_alias, {'type': 'openhands'}
        )
        # The previous token is NOT deleted by the store; the route does that.
        mock_delete_token.assert_not_called()

        member = await self._member_key(async_session_maker, org_id, user_id)
        assert member.llm_api_key.get_secret_value() == 'sk-new-managed-key'
        assert member.has_custom_llm_api_key is False

    @pytest.mark.asyncio
    async def test_rotate_managed_base_url_attaches_no_openhands_metadata(
        self, async_session_maker, managed_env
    ):
        """A non-openhands model on the managed base_url rotates without the
        openhands metadata marker.
        """
        user_id, org_id = await self._seed(
            async_session_maker,
            llm_model='anthropic/claude-sonnet-4',
            llm_base_url=self.MANAGED_BASE_URL,
        )

        with self._patched(async_session_maker) as (
            mock_delete_alias,
            mock_generate,
            _mock_delete_token,
        ):
            store = SaasSettingsStore(user_id, effective_org_id=org_id)
            rotation = await store.rotate_managed_llm_key()

        assert rotation.status == ManagedLlmKeyStatus.ROTATED
        assert rotation.openhands_type is False
        expected_alias = get_openhands_cloud_key_alias(user_id, str(org_id))
        mock_delete_alias.assert_awaited_once_with(key_alias=expected_alias)
        mock_generate.assert_awaited_once_with(
            user_id, str(org_id), expected_alias, None
        )

    @pytest.mark.asyncio
    async def test_rotate_non_managed_byok_base_url_is_rejected(
        self, async_session_maker, managed_env
    ):
        """A config pointing at a third-party base_url is NOT managed and is
        rejected before any key is generated or stored.
        """
        user_id, org_id = await self._seed(
            async_session_maker,
            llm_model='anthropic/claude-sonnet-4',
            llm_base_url='https://api.anthropic.com',
        )

        with self._patched(async_session_maker) as (
            mock_delete_alias,
            mock_generate,
            mock_delete_token,
        ):
            store = SaasSettingsStore(user_id, effective_org_id=org_id)
            rotation = await store.rotate_managed_llm_key()

        assert rotation.status == ManagedLlmKeyStatus.NOT_MANAGED
        mock_delete_alias.assert_not_called()
        mock_generate.assert_not_called()
        mock_delete_token.assert_not_called()

        member = await self._member_key(async_session_maker, org_id, user_id)
        assert member.llm_api_key.get_secret_value() == 'sk-old-managed-key'

    @pytest.mark.asyncio
    async def test_rotate_custom_byok_member_is_rejected(
        self, async_session_maker, managed_env
    ):
        """A member flagged BYOK (has_custom_llm_api_key) is rejected even when
        the effective model/base_url would otherwise be managed.
        """
        user_id, org_id = await self._seed(async_session_maker, member_custom=True)

        with self._patched(async_session_maker) as (
            mock_delete_alias,
            mock_generate,
            _mock_delete_token,
        ):
            store = SaasSettingsStore(user_id, effective_org_id=org_id)
            rotation = await store.rotate_managed_llm_key()

        assert rotation.status == ManagedLlmKeyStatus.BYOK
        mock_delete_alias.assert_not_called()
        mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_rotate_org_level_byok_key_is_rejected(
        self, async_session_maker, managed_env
    ):
        """An org-level key has precedence over the member managed key, so
        rotating the member key would not affect the effective runtime key.
        """
        user_id, org_id = await self._seed(async_session_maker, org_key='sk-org-byok')

        with self._patched(async_session_maker) as (
            mock_delete_alias,
            mock_generate,
            _mock_delete_token,
        ):
            store = SaasSettingsStore(user_id, effective_org_id=org_id)
            rotation = await store.rotate_managed_llm_key()
            current_key = await store.get_current_managed_llm_key()

        assert rotation.status == ManagedLlmKeyStatus.BYOK
        assert current_key is None
        mock_delete_alias.assert_not_called()
        mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_rotate_missing_member_does_not_persist_or_delete_old_key(
        self, async_session_maker, managed_env, session_maker
    ):
        """If the member row is gone at persist time, rotation reports
        MISSING_MEMBER, generates nothing, and does not delete the old key.
        """
        user_id, org_id = await self._seed(async_session_maker)
        # Drop the membership so load() resolves no acting member.
        with session_maker() as sync_session:
            from sqlalchemy import delete as sa_delete

            sync_session.execute(
                sa_delete(OrgMember).where(
                    OrgMember.org_id == org_id,
                    OrgMember.user_id == uuid.UUID(user_id),
                )
            )
            sync_session.commit()

        with self._patched(async_session_maker) as (
            mock_delete_alias,
            mock_generate,
            mock_delete_token,
        ):
            store = SaasSettingsStore(user_id, effective_org_id=org_id)
            rotation = await store.rotate_managed_llm_key()

        assert rotation.status == ManagedLlmKeyStatus.MISSING_MEMBER
        assert rotation.old_key is None
        assert rotation.new_key is None
        mock_delete_alias.assert_not_called()
        mock_generate.assert_not_called()
        mock_delete_token.assert_not_called()

    # --- endpoint wiring (delegates to the real store) ---

    @pytest.mark.asyncio
    async def test_route_refreshes_and_deletes_previous_token(
        self, async_session_maker, managed_env
    ):
        """The endpoint delegates to the real store and best-effort deletes the
        old token only after the new key is persisted.
        """
        user_id, org_id = await self._seed(async_session_maker)

        with self._patched_route(async_session_maker, user_id, org_id) as (
            mock_delete_alias,
            mock_generate,
            mock_delete_token,
        ):
            result = await refresh_managed_llm_api_key(
                user_id=user_id, effective_org_id=org_id
            )

        assert result == ManagedLlmApiKeyRefreshResponse(refreshed=True)
        mock_delete_alias.assert_awaited_once()
        mock_generate.assert_awaited_once()
        # The old token is deleted best-effort after persist.
        mock_delete_token.assert_awaited_once_with('sk-old-managed-key')

    @pytest.mark.asyncio
    async def test_route_rejects_non_managed_effective_config(
        self, async_session_maker, managed_env
    ):
        user_id, org_id = await self._seed(
            async_session_maker,
            llm_model='anthropic/claude-sonnet-4',
            llm_base_url='https://api.anthropic.com',
        )

        with self._patched_route(async_session_maker, user_id, org_id) as (
            mock_delete_alias,
            mock_generate,
            mock_delete_token,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await refresh_managed_llm_api_key(
                    user_id=user_id, effective_org_id=org_id
                )

        assert exc_info.value.status_code == 400
        assert 'non-managed LLM API key' in exc_info.value.detail
        mock_delete_alias.assert_not_called()
        mock_generate.assert_not_called()
        mock_delete_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_custom_byok_member_returns_400(
        self, async_session_maker, managed_env
    ):
        user_id, org_id = await self._seed(async_session_maker, member_custom=True)

        with self._patched_route(async_session_maker, user_id, org_id) as (
            mock_delete_alias,
            mock_generate,
            _mock_delete_token,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await refresh_managed_llm_api_key(
                    user_id=user_id, effective_org_id=org_id
                )

        assert exc_info.value.status_code == 400
        assert 'custom BYOK' in exc_info.value.detail
        mock_delete_alias.assert_not_called()
        mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_missing_member_returns_404_without_rotating(
        self, async_session_maker, managed_env
    ):
        """User belongs to no member row for the effective org."""
        user_id, _org_id = await self._seed(async_session_maker)
        other_org_id = uuid.uuid4()
        async with async_session_maker() as session:
            session.add(
                Org(
                    id=other_org_id,
                    name=f'other-org-{other_org_id}',
                    org_version=ORG_SETTINGS_VERSION,
                    agent_settings={'llm': {'model': 'openhands/claude-sonnet-4'}},
                )
            )
            await session.commit()

        with self._patched_route(async_session_maker, user_id, other_org_id) as (
            mock_delete_alias,
            mock_generate,
            mock_delete_token,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await refresh_managed_llm_api_key(
                    user_id=user_id, effective_org_id=other_org_id
                )

        assert exc_info.value.status_code == 404
        mock_delete_alias.assert_not_called()
        mock_generate.assert_not_called()
        mock_delete_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_continues_when_old_key_delete_fails(
        self, async_session_maker, managed_env
    ):
        user_id, org_id = await self._seed(async_session_maker)

        with (
            self._patched_route(async_session_maker, user_id, org_id) as (
                _mock_delete_alias,
                _mock_generate,
                _mock_delete_token_ok,
            ),
            patch(
                'storage.lite_llm_manager.LiteLlmManager.delete_key',
                new_callable=AsyncMock,
                side_effect=Exception('delete failed'),
            ) as mock_delete_token,
        ):
            result = await refresh_managed_llm_api_key(
                user_id=user_id, effective_org_id=org_id
            )

        assert result == ManagedLlmApiKeyRefreshResponse(refreshed=True)
        mock_delete_token.assert_awaited_once_with('sk-old-managed-key')

    @pytest.mark.asyncio
    async def test_route_unexpected_error_returns_500(self, managed_env):
        user_id = str(uuid.uuid4())
        org_id = uuid.uuid4()
        mock_store = MagicMock()
        mock_store.rotate_managed_llm_key = AsyncMock(side_effect=RuntimeError('boom'))
        with patch(
            'storage.saas_settings_store.SaasSettingsStore.get_instance',
            new_callable=AsyncMock,
            return_value=mock_store,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await refresh_managed_llm_api_key(
                    user_id=user_id, effective_org_id=org_id
                )

        assert exc_info.value.status_code == 500
        assert 'Failed to refresh managed LLM API key' in exc_info.value.detail


class TestDeleteByorKeyFromLitellm:
    """Test the delete_byor_key_from_litellm function with alias cleanup."""

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LiteLlmManager.delete_key')
    async def test_delete_constructs_alias_from_org(self, mock_delete_key):
        """Test that delete_byor_key_from_litellm builds the key alias from the effective org."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        byor_key = 'sk-byor-key-to-delete'
        expected_alias = BYOR_KEY_ALIAS_PATTERN.format(
            user_id=user_id, org_id=str(org_id)
        )
        mock_delete_key.return_value = None

        # Act
        result = await delete_byor_key_from_litellm(user_id, org_id, byor_key)

        # Assert
        assert result is True
        mock_delete_key.assert_called_once_with(byor_key, key_alias=expected_alias)

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.BYOR_KEY_ALIAS_PATTERN', 'env/{org_id}/u={user_id}')
    @patch('storage.lite_llm_manager.LiteLlmManager.delete_key')
    async def test_delete_uses_env_override_alias(self, mock_delete_key):
        """An overridden BYOR_KEY_ALIAS_PATTERN flows through to delete_key.

        Patches the imported constant in the api_keys module since the env
        var is read once at import time (see constants.py).
        """
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        byor_key = 'sk-byor-key-to-delete'
        mock_delete_key.return_value = None
        expected_alias = f'env/{org_id}/u={user_id}'

        # Act
        result = await delete_byor_key_from_litellm(user_id, org_id, byor_key)

        # Assert
        assert result is True
        mock_delete_key.assert_called_once_with(byor_key, key_alias=expected_alias)

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LiteLlmManager.delete_key')
    async def test_delete_returns_false_on_exception(self, mock_delete_key):
        """Test that exceptions during deletion return False."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        byor_key = 'sk-byor-key-to-delete'
        mock_delete_key.side_effect = Exception('LiteLLM API error')

        # Act
        result = await delete_byor_key_from_litellm(user_id, org_id, byor_key)

        # Assert
        assert result is False


class TestCreateByorKeyAlias:
    """Test the _create_byor_key_alias helper."""

    def test_uses_default_pattern(self):
        """Default pattern formats user_id and org_id with the literal defaults."""
        # Arrange
        user_id = 'user-123'
        org_id_str = str(uuid.uuid4())
        expected = BYOR_KEY_ALIAS_PATTERN.format(user_id=user_id, org_id=org_id_str)

        # Act
        result = _create_byor_key_alias(user_id, org_id_str)

        # Assert
        assert result == expected
        # The default should keep the production literal that LiteLLM
        # currently has provisioned keys tagged with.
        assert 'BYOR Key' in result
        assert user_id in result
        assert org_id_str in result

    def test_respects_env_override_pattern(self):
        """An overridden BYOR_KEY_ALIAS_PATTERN flows through the helper.

        Patches the imported constant in the api_keys module since it's
        read once at import time (mirrors the env-var read in constants.py).
        """
        custom_pattern = 'env/{org_id}/u={user_id}'
        with patch('server.routes.api_keys.BYOR_KEY_ALIAS_PATTERN', custom_pattern):
            result = _create_byor_key_alias('user-9', 'org-7')

        assert result == 'env/org-7/u=user-9'

    def test_missing_placeholder_raises_keyerror(self):
        """A pattern with an unexpected placeholder fails loudly (str.format KeyError)."""
        bad_pattern = '{environment}/{something_else}'
        with patch('server.routes.api_keys.BYOR_KEY_ALIAS_PATTERN', bad_pattern):
            with pytest.raises(KeyError):
                _create_byor_key_alias('user-9', 'org-7')


class TestGenerateByorKey:
    """Test the generate_byor_key function."""

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LiteLlmManager.generate_key')
    async def test_passes_default_alias_and_team_id_to_litellm(self, mock_generate_key):
        """generate_byor_key builds the alias from the helper and passes org_id as str."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        org_id_str = str(org_id)
        new_key = 'sk-new-generated-key'
        mock_generate_key.return_value = new_key
        expected_alias = BYOR_KEY_ALIAS_PATTERN.format(
            user_id=user_id, org_id=org_id_str
        )

        # Act
        result = await generate_byor_key(user_id, org_id)

        # Assert
        assert result == new_key
        mock_generate_key.assert_called_once_with(
            user_id,
            org_id_str,
            expected_alias,
            {'type': 'byor'},
        )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LiteLlmManager.generate_key')
    async def test_uses_env_override_alias(self, mock_generate_key):
        """An overridden pattern flows through to LiteLLM.generate_key."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        mock_generate_key.return_value = 'sk-key'
        custom_pattern = 'custom/{org_id}/{user_id}'

        with patch('server.routes.api_keys.BYOR_KEY_ALIAS_PATTERN', custom_pattern):
            await generate_byor_key(user_id, org_id)

        # Assert
        mock_generate_key.assert_called_once_with(
            user_id,
            str(org_id),
            custom_pattern.format(user_id=user_id, org_id=str(org_id)),
            {'type': 'byor'},
        )

    @pytest.mark.asyncio
    @patch('storage.lite_llm_manager.LiteLlmManager.generate_key')
    async def test_returns_none_on_exception(self, mock_generate_key):
        """Exceptions during generation surface as None (caller handles 500)."""
        # Arrange
        mock_generate_key.side_effect = Exception('LiteLLM API error')

        # Act
        result = await generate_byor_key('user-123', uuid.uuid4())

        # Assert
        assert result is None


class TestCheckByorPermitted:
    """Test the check_byor_permitted endpoint."""

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    async def test_permitted_when_enabled(self, mock_check_enabled):
        """Test that permitted=True is returned when BYOR export is enabled."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        mock_check_enabled.return_value = True

        # Act
        result = await check_byor_permitted(user_id=user_id, effective_org_id=org_id)

        # Assert
        assert result == ByorPermittedResponse(permitted=True)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    async def test_not_permitted_when_disabled(self, mock_check_enabled):
        """Test that permitted=False is returned when BYOR export is disabled."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        mock_check_enabled.return_value = False

        # Act
        result = await check_byor_permitted(user_id=user_id, effective_org_id=org_id)

        # Assert
        assert result == ByorPermittedResponse(permitted=False)
        mock_check_enabled.assert_called_once_with(user_id, org_id=org_id)

    @pytest.mark.asyncio
    @patch('storage.org_service.OrgService.check_byor_export_enabled')
    async def test_error_raises_500(self, mock_check_enabled):
        """Test that an exception raises 500 error."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        mock_check_enabled.side_effect = Exception('Database error')

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await check_byor_permitted(user_id=user_id, effective_org_id=org_id)

        assert exc_info.value.status_code == 500
        assert 'Failed to check BYOR export permission' in exc_info.value.detail


class TestGetCurrentApiKey:
    """Test the get_current_api_key endpoint."""

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.get_user_auth')
    async def test_returns_api_key_info_for_bearer_auth(self, mock_get_user_auth):
        """Org-bound API key reports its bound org in both fields."""
        # Arrange
        user_id = 'user-123'
        org_id = uuid.uuid4()
        mock_request = MagicMock()

        user_auth = SaasUserAuth(
            refresh_token=SecretStr('mock-token'),
            user_id=user_id,
            auth_type=AuthType.BEARER,
            api_key_org_id=org_id,
            api_key_id=42,
            api_key_name='My Production Key',
        )
        # Bound keys resolve to their bound org in all contexts.
        user_auth.get_effective_org_id = AsyncMock(return_value=org_id)
        mock_get_user_auth.return_value = user_auth

        # Act
        result = await get_current_api_key(request=mock_request, user_id=user_id)

        # Assert
        assert isinstance(result, CurrentApiKeyResponse)
        assert result.org_id == str(org_id)
        assert result.bound_org_id == str(org_id)
        assert result.id == 42
        assert result.name == 'My Production Key'
        assert result.user_id == user_id
        assert result.auth_type == 'bearer'

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.get_user_auth')
    async def test_returns_400_for_cookie_auth(self, mock_get_user_auth):
        """Test that 400 Bad Request is returned when using cookie authentication."""
        # Arrange
        user_id = 'user-123'
        mock_request = MagicMock()

        mock_user_auth = MagicMock()
        mock_user_auth.get_auth_type.return_value = AuthType.COOKIE
        mock_get_user_auth.return_value = mock_user_auth

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_current_api_key(request=mock_request, user_id=user_id)

        assert exc_info.value.status_code == 400
        assert 'API key authentication' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('server.routes.api_keys.get_user_auth')
    async def test_unbound_key_reports_effective_org(self, mock_get_user_auth):
        """An unbound API key reports the resolved effective org and None bound."""
        # Arrange
        user_id = 'user-123'
        effective_org = uuid.uuid4()
        mock_request = MagicMock()

        user_auth = SaasUserAuth(
            refresh_token=SecretStr('mock-token'),
            user_id=user_id,
            auth_type=AuthType.BEARER,
            api_key_org_id=None,  # Unbound key
            api_key_id=42,
            api_key_name='Multi-org Key',
        )
        # Unbound keys resolve to the request's effective org (X-Org-Id or
        # user.current_org_id).
        user_auth.get_effective_org_id = AsyncMock(return_value=effective_org)
        mock_get_user_auth.return_value = user_auth

        # Act
        result = await get_current_api_key(request=mock_request, user_id=user_id)

        # Assert
        assert isinstance(result, CurrentApiKeyResponse)
        assert result.org_id == str(effective_org)
        assert result.bound_org_id is None
        assert result.id == 42
        assert result.name == 'Multi-org Key'


class TestApiKeyCreateValidation:
    """Test the ApiKeyCreate Pydantic model's validators."""

    def test_accepts_no_window(self):
        """A request with neither bound set is valid."""
        from server.routes.api_keys import ApiKeyCreate

        model = ApiKeyCreate(name='unbound-style')
        assert model.not_before is None
        assert model.expires_at is None
        assert model.org_id is None

    def test_accepts_explicit_unbound_org_id(self):
        """An explicit ``org_id=None`` is preserved as a model-field-set signal."""
        from server.routes.api_keys import ApiKeyCreate

        model = ApiKeyCreate(name='unbound-style', org_id=None)
        assert model.org_id is None
        # ``org_id`` is in ``model_fields_set`` even when set to ``None``;
        # the route uses this to distinguish "explicit unbound" from
        # "field omitted -> fall back to effective org".
        assert 'org_id' in model.model_fields_set

    def test_accepts_specific_org_id(self):
        """An explicit ``org_id=<UUID>`` binds the new key to that org."""
        from server.routes.api_keys import ApiKeyCreate

        bound_org = uuid.uuid4()
        model = ApiKeyCreate(name='bound-style', org_id=bound_org)
        assert model.org_id == bound_org

    def test_accepts_not_before_only(self):
        """A request with only not_before is valid."""
        from server.routes.api_keys import ApiKeyCreate

        future = datetime.now(UTC) + timedelta(days=1)
        model = ApiKeyCreate(name='future-key', not_before=future)
        assert model.not_before == future
        assert model.expires_at is None

    def test_accepts_expires_at_only(self):
        """A request with only expires_at is valid."""
        from server.routes.api_keys import ApiKeyCreate

        future = datetime.now(UTC) + timedelta(days=1)
        model = ApiKeyCreate(name='expiring-key', expires_at=future)
        assert model.not_before is None
        assert model.expires_at == future

    def test_accepts_valid_window(self):
        """A request with not_before < expires_at is valid."""
        from server.routes.api_keys import ApiKeyCreate

        not_before = datetime.now(UTC) + timedelta(days=1)
        expires_at = not_before + timedelta(days=30)
        model = ApiKeyCreate(
            name='window-key', not_before=not_before, expires_at=expires_at
        )
        assert model.not_before == not_before
        assert model.expires_at == expires_at

    def test_rejects_expires_at_in_past(self):
        from pydantic import ValidationError
        from server.routes.api_keys import ApiKeyCreate

        with pytest.raises(ValidationError) as exc_info:
            ApiKeyCreate(
                name='past-key',
                expires_at=datetime.now(UTC) - timedelta(days=1),
            )
        assert 'Expiration' in str(exc_info.value)

    def test_rejects_inverted_window(self):
        from pydantic import ValidationError
        from server.routes.api_keys import ApiKeyCreate

        not_before = datetime.now(UTC) + timedelta(days=10)
        expires_at = datetime.now(UTC) + timedelta(days=1)
        with pytest.raises(ValidationError) as exc_info:
            ApiKeyCreate(
                name='inverted-key',
                not_before=not_before,
                expires_at=expires_at,
            )
        assert 'not_before must be earlier than expires_at' in str(exc_info.value)

    def test_rejects_equal_window(self):
        """not_before == expires_at is rejected (degenerate window)."""
        from pydantic import ValidationError
        from server.routes.api_keys import ApiKeyCreate

        same = datetime.now(UTC) + timedelta(days=1)
        with pytest.raises(ValidationError) as exc_info:
            ApiKeyCreate(
                name='equal-key',
                not_before=same,
                expires_at=same,
            )
        assert 'not_before must be earlier than expires_at' in str(exc_info.value)


class TestCreateApiKeyRoute:
    """End-to-end tests for the ``POST /api/keys`` route."""

    @pytest.mark.asyncio
    async def test_unbound_org_id_creates_unbound_key(self):
        """Regression: ``org_id=None`` must produce an unbound (org_id NULL) row.

        The route previously delegated to ``ApiKeyStore.create_api_key``
        with ``org_id=None``; the store then silently rebound to
        ``user.current_org_id``, so the row inserted never matched the
        route's ``name + org_id is None`` lookup and the route returned
        500. The store now accepts ``use_current_org_fallback=False`` and
        the route passes that flag.
        """
        from server.routes.api_keys import ApiKeyCreate, create_api_key

        # A row the route will "find" after insert: an unbound key with
        # the matching name. This is what the (now fixed) route expects
        # ``list_api_keys`` to surface.
        matching_key = MagicMock()
        matching_key.id = 1
        matching_key.name = 'AllOrgs'
        matching_key.org_id = None
        matching_key.created_at = datetime.now(UTC)
        matching_key.last_used_at = None
        matching_key.not_before = None
        matching_key.expires_at = None

        captured: dict = {}

        async def fake_create_api_key(
            user_id,
            name,
            expires_at=None,
            not_before=None,
            org_id=...,
            **kwargs,
        ):
            # The store should be called with the route's explicit
            # ``None`` and the fallback disabled.
            captured['org_id'] = org_id
            captured['use_current_org_fallback'] = kwargs.get(
                'use_current_org_fallback'
            )
            return 'sk-oh-fake'

        with (
            patch(
                'server.routes.api_keys.api_key_store.create_api_key',
                AsyncMock(side_effect=fake_create_api_key),
            ),
            patch(
                'server.routes.api_keys.api_key_store.list_api_keys',
                AsyncMock(return_value=[matching_key]),
            ),
        ):
            result = await create_api_key(
                key_data=ApiKeyCreate(name='AllOrgs', org_id=None),
                user_id=str(uuid.uuid4()),
                effective_org_id=uuid.uuid4(),
            )

        assert captured['org_id'] is None
        assert captured['use_current_org_fallback'] is False
        assert result.org_id is None
        assert result.name == 'AllOrgs'

    @pytest.mark.asyncio
    async def test_omitted_org_id_falls_back_to_effective_org(self):
        """An *omitted* ``org_id`` binds the new key to the effective org."""
        from server.routes.api_keys import ApiKeyCreate, create_api_key

        effective_org = uuid.uuid4()
        bound_key = MagicMock()
        bound_key.id = 2
        bound_key.name = 'Bound'
        bound_key.org_id = effective_org
        bound_key.created_at = datetime.now(UTC)
        bound_key.last_used_at = None
        bound_key.not_before = None
        bound_key.expires_at = None

        captured: dict = {}

        async def fake_create_api_key(
            user_id,
            name,
            expires_at=None,
            not_before=None,
            org_id=...,
            **kwargs,
        ):
            captured['org_id'] = org_id
            captured['use_current_org_fallback'] = kwargs.get(
                'use_current_org_fallback'
            )
            return 'sk-oh-fake'

        with (
            patch(
                'server.routes.api_keys.api_key_store.create_api_key',
                AsyncMock(side_effect=fake_create_api_key),
            ),
            patch(
                'server.routes.api_keys.api_key_store.list_api_keys',
                AsyncMock(return_value=[bound_key]),
            ),
            patch(
                # Membership check on the effective org -- return a non-None
                # member so the route passes the check.
                'server.routes.api_keys.OrgMemberStore.get_org_member',
                AsyncMock(return_value=MagicMock()),
            ),
        ):
            result = await create_api_key(
                key_data=ApiKeyCreate(name='Bound'),  # no org_id
                user_id=str(uuid.uuid4()),
                effective_org_id=effective_org,
            )

        # Route forwards the effective org to the store, and disables the
        # fallback so the store doesn't double-rebind.
        assert captured['org_id'] == effective_org
        assert captured['use_current_org_fallback'] is False
        assert result.org_id == effective_org
