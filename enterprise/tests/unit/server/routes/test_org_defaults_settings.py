"""Unit tests for organization-default settings models and serialization."""

from unittest.mock import MagicMock

from pydantic import SecretStr
from server.constants import LITE_LLM_API_URL
from server.routes.org_models import (
    MASKED_API_KEY,
    OrgDefaultsSettingsResponse,
    OrgUpdate,
)
from storage.org import Org


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


def test_from_org_denormalizes_litellm_proxy_prefix_and_returns_base_url_as_stored():
    """Managed-model responses should be denormalized for the frontend."""
    org = MagicMock(spec=Org)
    org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'litellm_proxy/minimax-m2.5',
            'base_url': LITE_LLM_API_URL,
            'api_key': MASKED_API_KEY,
        },
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None

    response = OrgDefaultsSettingsResponse.from_org(org)

    assert response.agent_settings.llm.model == 'openhands/minimax-m2.5'
    assert response.agent_settings.llm.base_url == LITE_LLM_API_URL
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
