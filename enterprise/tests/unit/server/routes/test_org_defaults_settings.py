"""Unit tests for organization-default settings models and serialization."""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr, ValidationError
from server.constants import LITE_LLM_API_URL
from server.routes.org_models import (
    MASKED_API_KEY,
    OrgAppSettingsUpdate,
    OrgDefaultsSettingsResponse,
    OrgUpdate,
)
from storage.org import Org

from openhands.app_server.settings.settings_models import MarketplaceRegistration
from openhands.sdk.settings import ACPAgentSettings


def test_org_update_keeps_sparse_diff_dicts():
    """OrgUpdate should preserve sparse org-default diffs as dictionaries."""
    update_data = OrgUpdate.model_validate(
        {
            'agent_settings_diff': {'llm': {'model': 'claude-3-5-sonnet'}},
            'conversation_settings_diff': {'security_analyzer': 'llm'},
        }
    )

    assert update_data.agent_settings_diff == {'llm': {'model': 'claude-3-5-sonnet'}}
    assert update_data.conversation_settings_diff == {'security_analyzer': 'llm'}


def test_normalize_agent_settings_masks_api_key_in_json_on_empty_and_real_keys():
    """Nested api_key values are lifted and masked in the JSON patch."""
    real_key = OrgUpdate.model_validate(
        {'agent_settings_diff': {'llm': {'model': 'anthropic/x', 'api_key': 'sk-raw'}}}
    )
    empty_key = OrgUpdate.model_validate(
        {
            'agent_settings_diff': {
                'llm': {'model': 'openhands/x', 'api_key': '', 'base_url': None},
            },
        }
    )

    assert real_key.llm_api_key == 'sk-raw'
    assert real_key.agent_settings_diff is not None
    assert real_key.agent_settings_diff['llm']['api_key'] == MASKED_API_KEY
    assert empty_key.llm_api_key == ''
    assert empty_key.agent_settings_diff is not None
    assert empty_key.agent_settings_diff['llm']['api_key'] == MASKED_API_KEY


def test_normalize_agent_settings_fills_base_url_for_all_providers():
    """Managed and BYOR providers should keep usable base URLs in diffs."""
    openhands_null = OrgUpdate.model_validate(
        {
            'agent_settings_diff': {
                'llm': {'model': 'openhands/claude-3', 'base_url': None},
            },
        }
    )
    openhands_missing = OrgUpdate.model_validate(
        {'agent_settings_diff': {'llm': {'model': 'openhands/claude-3'}}}
    )
    anthropic_null = OrgUpdate.model_validate(
        {
            'agent_settings_diff': {
                'llm': {'model': 'anthropic/claude-3-opus-20240229', 'base_url': None},
            },
        }
    )

    openhands_null_diff = openhands_null.agent_settings_diff
    assert openhands_null_diff is not None
    assert openhands_null_diff['llm']['model'] == 'openhands/claude-3'
    assert openhands_null_diff['llm']['base_url'].rstrip('/') == (
        LITE_LLM_API_URL.rstrip('/')
    )

    openhands_missing_diff = openhands_missing.agent_settings_diff
    assert openhands_missing_diff is not None
    assert openhands_missing_diff['llm']['model'] == 'openhands/claude-3'
    assert openhands_missing_diff['llm']['base_url'].rstrip('/') == (
        LITE_LLM_API_URL.rstrip('/')
    )

    anthropic_diff = anthropic_null.agent_settings_diff
    assert anthropic_diff is not None
    anthropic_base = anthropic_diff['llm']['base_url']
    assert isinstance(anthropic_base, str)
    assert 'anthropic.com' in anthropic_base


def test_from_org_validates_persisted_openhands_agent_kind():
    """GIVEN: An org row whose persisted ``agent_settings`` carry the
        canonical ``agent_kind: 'openhands'`` discriminator (the exact shape
        from the 500-error log)
    WHEN: ``OrgDefaultsSettingsResponse.from_org`` serializes the org
    THEN: The response is built without a Pydantic literal-mismatch error
        and exposes the expected canonical agent kind and llm model.
    """
    # Arrange
    org = MagicMock(spec=Org)
    org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'agent_kind': 'openhands',
        'llm': {'model': 'openhands/claude', 'base_url': LITE_LLM_API_URL},
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None

    # Act
    response = OrgDefaultsSettingsResponse.from_org(org)

    # Assert
    assert response.agent_settings.agent_kind == 'openhands'
    assert response.agent_settings.llm.model == 'openhands/claude'


def test_from_org_preserves_acp_agent_settings_without_500():
    """GIVEN: An org on ACP — persisted ``agent_kind: 'acp'`` with a null
        ``agent_context`` (the exact shape behind the /api/organizations 500s).
    WHEN: ``OrgDefaultsSettingsResponse.from_org`` serializes the org.
    THEN: It returns the ``ACPAgentSettings`` variant instead of force-casting
        to ``OpenHandsAgentSettings`` (which 500'd on the non-nullable
        ``agent_context``).
    """
    org = MagicMock(spec=Org)
    org.agent_settings = {
        'agent_kind': 'acp',
        'acp_server': 'claude-code',
        'llm': {'model': 'litellm_proxy/anthropic/claude-sonnet-4'},
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None

    response = OrgDefaultsSettingsResponse.from_org(org)

    assert isinstance(response.agent_settings, ACPAgentSettings)
    assert response.agent_settings.agent_kind == 'acp'
    assert response.agent_settings.agent_context is None


def test_from_org_keeps_openhands_prefix_and_hides_managed_base_url():
    """Managed OpenHands models should return the public prefix in basic mode."""
    org = MagicMock(spec=Org)
    org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'openhands/minimax-m2.5',
            'base_url': LITE_LLM_API_URL,
            'api_key': MASKED_API_KEY,
        },
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None

    response = OrgDefaultsSettingsResponse.from_org(org)

    assert response.agent_settings.llm.model == 'openhands/minimax-m2.5'
    assert response.agent_settings.llm.base_url is None
    assert response.agent_settings.llm.api_key is None


def test_from_org_returns_provider_default_base_url_as_stored_for_non_managed_models():
    """BYOR provider-default base URLs should round-trip unchanged."""
    from openhands.app_server.utils.llm import get_provider_api_base as _provider_base

    anthropic_default = _provider_base('anthropic/claude-3-opus-20240229')
    assert anthropic_default is not None

    org = MagicMock(spec=Org)
    org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'anthropic/claude-3-opus-20240229',
            'base_url': anthropic_default,
        },
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None

    response = OrgDefaultsSettingsResponse.from_org(org)

    assert response.agent_settings.llm.model == 'anthropic/claude-3-opus-20240229'
    assert response.agent_settings.llm.base_url == anthropic_default


def test_from_org_keeps_custom_base_url_that_is_not_provider_default():
    """Custom BYOR base URLs should be preserved in the wrapper response."""
    org = MagicMock(spec=Org)
    org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'anthropic/claude-3-opus-20240229',
            'base_url': 'https://company-proxy.internal/anthropic',
        },
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = SecretStr('search-key-1234')

    response = OrgDefaultsSettingsResponse.from_org(org)

    assert (
        response.agent_settings.llm.base_url
        == 'https://company-proxy.internal/anthropic'
    )
    assert response.search_api_key == '****1234'


# --- Tests for OrgAppSettingsUpdate registered_marketplaces ---


class TestOrgAppSettingsUpdateMarketplaceValidation:
    """Tests for registered_marketplaces in OrgAppSettingsUpdate."""

    def test_valid_marketplace_list(self):
        """Test that OrgAppSettingsUpdate accepts valid marketplace list."""
        update = OrgAppSettingsUpdate(
            registered_marketplaces=[
                MarketplaceRegistration(name='test', source='github:owner/repo')
            ]
        )
        assert len(update.registered_marketplaces) == 1
        assert update.registered_marketplaces[0].name == 'test'
        assert update.registered_marketplaces[0].source == 'github:owner/repo'

    def test_null_marketplaces_allowed(self):
        """Test that None is allowed for PATCH semantics (field not updated)."""
        update = OrgAppSettingsUpdate()
        assert update.registered_marketplaces is None

    def test_empty_list_valid(self):
        """Test that empty list is valid (explicitly clear marketplaces)."""
        update = OrgAppSettingsUpdate(registered_marketplaces=[])
        assert update.registered_marketplaces == []

    def test_invalid_marketplace_rejected(self):
        """Test that invalid marketplace source is rejected."""
        with pytest.raises(ValidationError):
            OrgAppSettingsUpdate(
                registered_marketplaces=[{'name': 'test', 'source': 'invalid!!source'}]
            )

    def test_multiple_valid_marketplaces(self):
        """Test that multiple valid marketplaces are accepted."""
        update = OrgAppSettingsUpdate(
            registered_marketplaces=[
                MarketplaceRegistration(
                    name='marketplace-1',
                    source='github:owner/repo1',
                    auto_load=True,
                ),
                MarketplaceRegistration(
                    name='marketplace-2',
                    source='github:owner/repo2',
                ),
            ]
        )
        assert len(update.registered_marketplaces) == 2

    def test_marketplace_with_all_fields(self):
        """Test marketplace with all optional fields."""
        update = OrgAppSettingsUpdate(
            registered_marketplaces=[
                MarketplaceRegistration(
                    name='full-featured',
                    source='github:owner/repo',
                    ref='v1.0.0',
                    repo_path='marketplaces/plugins',
                    auto_load=True,
                ),
            ]
        )
        assert update.registered_marketplaces[0].name == 'full-featured'
        assert update.registered_marketplaces[0].ref == 'v1.0.0'
        assert update.registered_marketplaces[0].repo_path == 'marketplaces/plugins'
        assert update.registered_marketplaces[0].auto_load is True


class TestUpdateOrgAppSettingsRoute:
    """Route-level behavior for the POST /orgs/app handler."""

    @pytest.mark.asyncio
    async def test_concurrent_modification_maps_to_409(self):
        """A concurrent-modification conflict surfaces as HTTP 409, not 500."""
        from datetime import datetime, timezone
        from unittest.mock import AsyncMock

        from fastapi import HTTPException
        from server.routes.org_models import OrgConcurrentModificationError
        from server.routes.orgs import update_org_app_settings

        # Arrange - service raises the conflict; no marketplaces => no admin gate.
        now = datetime.now(timezone.utc)
        service = MagicMock()
        service.update_org_app_settings = AsyncMock(
            side_effect=OrgConcurrentModificationError(
                org_id='o1', expected_version=now, actual_version=now
            )
        )
        update_data = OrgAppSettingsUpdate(enable_proactive_conversation_starters=False)

        # Act & Assert
        with pytest.raises(HTTPException) as exc:
            await update_org_app_settings(
                update_data, MagicMock(), service=service, user_id='u1'
            )
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_marketplace_edit_requires_admin(self, monkeypatch):
        """Editing org marketplaces without EDIT_ORG_SETTINGS is blocked (403)."""
        from unittest.mock import AsyncMock

        import server.routes.orgs as orgs_module
        from fastapi import HTTPException
        from server.routes.orgs import update_org_app_settings

        # Arrange - deny the marketplace permission; the write must not happen.
        async def _deny(request, user_id, permission):
            raise HTTPException(status_code=403, detail='forbidden')

        monkeypatch.setattr(orgs_module, 'authorize_permission', _deny)
        service = MagicMock()
        service.update_org_app_settings = AsyncMock()
        update_data = OrgAppSettingsUpdate(
            registered_marketplaces=[
                MarketplaceRegistration(name='team', source='github:o/team')
            ]
        )

        # Act & Assert
        with pytest.raises(HTTPException) as exc:
            await update_org_app_settings(
                update_data, MagicMock(), service=service, user_id='u1'
            )
        assert exc.value.status_code == 403
        service.update_org_app_settings.assert_not_called()
