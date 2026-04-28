"""Tests for SDK settings compatibility shim."""

import pytest

from openhands.utils.sdk_settings_compat import (
    _HAS_DISCRIMINATED_UNION,
    ACPAgentSettings,
    AgentSettingsConfig,
    LLMAgentSettings,
    default_agent_settings,
    export_agent_settings_schema,
    validate_agent_settings,
)


def test_has_discriminated_union_is_bool():
    assert isinstance(_HAS_DISCRIMINATED_UNION, bool)


def test_default_agent_settings_returns_instance():
    settings = default_agent_settings()
    assert settings is not None


def test_validate_agent_settings_with_llm_dict():
    """validate_agent_settings should accept a plain dict and return an agent settings object."""
    result = validate_agent_settings({})
    assert result is not None


def test_validate_agent_settings_with_acp_kind_raises_or_returns():
    """validate_agent_settings with kind='acp' either returns ACP settings or raises RuntimeError."""
    data = {'kind': 'acp', 'agent_kind': 'acp'}
    if _HAS_DISCRIMINATED_UNION:
        # With SDK support, kind='acp' produces an ACPAgentSettings
        result = validate_agent_settings(data)
        assert isinstance(result, ACPAgentSettings)
    else:
        # Without SDK support, kind='acp' raises RuntimeError
        with pytest.raises(RuntimeError, match='kind=.acp.'):
            validate_agent_settings(data)


def test_export_agent_settings_schema_returns_schema():
    schema = export_agent_settings_schema()
    assert schema is not None
    # Schema should have a model_dump method (Pydantic model)
    dumped = schema.model_dump(mode='json')
    assert isinstance(dumped, dict)


def test_llm_agent_settings_is_usable():
    """LLMAgentSettings should be constructable."""
    settings = LLMAgentSettings()
    assert settings is not None


@pytest.mark.skipif(
    not _HAS_DISCRIMINATED_UNION,
    reason='Discriminated union not available in installed SDK',
)
def test_acp_agent_settings_is_usable():
    """ACPAgentSettings should be constructable when SDK supports it."""
    settings = ACPAgentSettings()
    assert settings is not None
    assert hasattr(settings, 'agent_kind')
    assert settings.agent_kind == 'acp'


def test_agent_settings_config_is_type():
    """AgentSettingsConfig should be a usable type."""
    assert AgentSettingsConfig is not None
