"""Unit tests for OrgLLMSettingsService.

Tests the service layer for organization LLM settings operations.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from server.constants import LITE_LLM_API_URL
from server.routes.org_models import (
    MASKED_API_KEY,
    OrgLLMSettingsResponse,
    OrgLLMSettingsUpdate,
    OrgNotFoundError,
    OrgUpdate,
)
from server.services import org_llm_settings_service as service_module
from server.services.org_llm_settings_service import OrgLLMSettingsService
from storage.org import Org

from openhands.sdk.settings import AgentSettings, ConversationSettings


def test_org_update_accepts_typed_settings_objects():
    """OrgUpdate should parse wire payloads into typed settings objects."""
    update_data = OrgUpdate.model_validate(
        {
            "agent_settings": {"llm": {"model": "claude-3-5-sonnet"}},
            "conversation_settings": {"security_analyzer": "llm"},
        }
    )

    assert isinstance(update_data.agent_settings, AgentSettings)
    assert update_data.agent_settings_patch() == {"llm": {"model": "claude-3-5-sonnet"}}
    assert isinstance(update_data.conversation_settings, ConversationSettings)
    assert update_data.conversation_settings_patch() == {"security_analyzer": "llm"}


@pytest.fixture
def user_id():
    """Create a test user ID."""
    return str(uuid.uuid4())


@pytest.fixture
def org_id():
    """Create a test org ID."""
    return uuid.uuid4()


@pytest.fixture
def mock_org(org_id):
    """Create a mock organization with LLM settings."""
    org = MagicMock(spec=Org)
    org.id = org_id
    org.agent_settings = {
        "schema_version": 1,
        "agent": "CodeActAgent",
        "llm": {
            "model": "claude-3",
            "base_url": "https://api.anthropic.com",
        },
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None
    return org


@pytest.fixture
def mock_store():
    """Create a mock OrgLLMSettingsStore."""
    return MagicMock()


@pytest.fixture
def mock_user_context(user_id):
    """Create a mock UserContext that returns the user_id."""
    context = MagicMock()
    context.get_user_id = AsyncMock(return_value=user_id)
    return context


@pytest.mark.asyncio
async def test_get_org_llm_settings_success(
    user_id, mock_org, mock_store, mock_user_context
):
    """GIVEN: A user with a current organization
    WHEN: get_org_llm_settings is called
    THEN: OrgLLMSettingsResponse is returned with correct data
    """
    # Arrange
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.get_org_llm_settings()

    # Assert
    assert isinstance(result, OrgLLMSettingsResponse)
    assert result.agent_settings.llm.model == "claude-3"
    assert result.agent_settings.agent == "CodeActAgent"
    mock_store.get_current_org_by_user_id.assert_called_once_with(user_id)


@pytest.mark.asyncio
async def test_get_org_llm_settings_user_not_authenticated(mock_store):
    """GIVEN: A user is not authenticated
    WHEN: get_org_llm_settings is called
    THEN: ValueError is raised
    """
    # Arrange
    mock_user_context = MagicMock()
    mock_user_context.get_user_id = AsyncMock(return_value=None)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        await service.get_org_llm_settings()

    assert "not authenticated" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_org_llm_settings_org_not_found(
    user_id, mock_store, mock_user_context
):
    """GIVEN: A user has no current organization
    WHEN: get_org_llm_settings is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=None)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(OrgNotFoundError) as exc_info:
        await service.get_org_llm_settings()

    assert "No current organization" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_org_llm_settings_success(
    user_id, mock_org, mock_store, mock_user_context, monkeypatch
):
    """Deprecated /llm writes should forward through OrgService using OrgUpdate."""
    updated_org = MagicMock(spec=Org)
    updated_org.id = mock_org.id
    updated_org.agent_settings = {
        "schema_version": 1,
        "agent": "CodeActAgent",
        "llm": {"model": "new-model"},
    }
    updated_org.conversation_settings = {
        "confirmation_mode": False,
        "max_iterations": 100,
    }
    updated_org.llm_api_key = None
    updated_org.search_api_key = None

    update_data = OrgUpdate(
        agent_settings={"llm": {"model": "new-model"}},
        conversation_settings={
            "confirmation_mode": False,
            "max_iterations": 100,
        },
    )

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    update_org_with_permissions = AsyncMock(return_value=updated_org)
    monkeypatch.setattr(
        service_module.OrgService,
        "update_org_with_permissions",
        update_org_with_permissions,
    )
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    result = await service.update_org_llm_settings(update_data)

    assert isinstance(result, OrgLLMSettingsResponse)
    assert result.agent_settings.llm.model == "new-model"
    assert result.conversation_settings.confirmation_mode is False
    assert result.conversation_settings.max_iterations == 100
    update_org_with_permissions.assert_awaited_once_with(
        org_id=mock_org.id,
        update_data=update_data,
        user_id=user_id,
    )


@pytest.mark.asyncio
async def test_update_org_llm_settings_no_changes(
    mock_org, mock_store, mock_user_context, monkeypatch
):
    """No-op deprecated /llm writes should return current settings immediately."""
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    update_org_with_permissions = AsyncMock()
    monkeypatch.setattr(
        service_module.OrgService,
        "update_org_with_permissions",
        update_org_with_permissions,
    )
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    result = await service.update_org_llm_settings(OrgUpdate())

    assert isinstance(result, OrgLLMSettingsResponse)
    assert result.agent_settings.llm.model == "claude-3"
    update_org_with_permissions.assert_not_called()


@pytest.mark.asyncio
async def test_update_org_llm_settings_org_not_found(
    user_id, mock_store, mock_user_context
):
    """GIVEN: A user has no current organization
    WHEN: update_org_llm_settings is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    update_data = OrgLLMSettingsUpdate(agent_settings={"llm": {"model": "new-model"}})

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=None)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(OrgNotFoundError) as exc_info:
        await service.update_org_llm_settings(update_data)

    assert "No current organization" in str(exc_info.value)


def test_normalize_agent_settings_masks_api_key_in_json_on_empty_and_real_keys():
    """GIVEN: Wire payloads that either carry a real raw api_key (BYOR save) or
           an empty string api_key (managed/OpenHands switch)
    WHEN:  OrgLLMSettingsUpdate's model validator runs
    THEN:  both shapes lift the raw value to ``llm_api_key`` for encrypted
           column sync AND leave the universal ``MASKED_API_KEY`` marker in
           ``agent_settings.llm.api_key``, so the three storage locations
           (``org._llm_api_key``, ``org.agent_settings.llm.api_key``,
           ``org_member.agent_settings_diff.llm.api_key``) stay in sync once
           the update is applied + propagated.
    """
    # Arrange + Act
    real_key = OrgLLMSettingsUpdate.model_validate(
        {"agent_settings": {"llm": {"model": "anthropic/x", "api_key": "sk-raw"}}}
    )
    empty_key = OrgLLMSettingsUpdate.model_validate(
        {
            "agent_settings": {
                "llm": {"model": "openhands/x", "api_key": "", "base_url": None},
            },
        }
    )

    # Assert — masked in JSON in both cases; lifted raw value on top-level.
    assert real_key.llm_api_key == "sk-raw"
    assert real_key.agent_settings_patch() is not None
    assert real_key.agent_settings_patch()["llm"]["api_key"] == MASKED_API_KEY
    assert empty_key.llm_api_key == ""
    assert empty_key.agent_settings_patch() is not None
    assert empty_key.agent_settings_patch()["llm"]["api_key"] == MASKED_API_KEY


def test_normalize_agent_settings_fills_base_url_for_all_providers():
    """GIVEN: Wire payloads from the basic view that send ``base_url: null`` for
           various providers (OpenHands managed + BYOR providers like OpenAI
           and Anthropic)
    WHEN:  OrgLLMSettingsUpdate's model validator runs
    THEN:  ``base_url`` is populated for every recognised provider —
           ``LITE_LLM_API_URL`` for OpenHands/managed models and the
           litellm-derived default URL for non-managed providers (via
           ``get_provider_api_base``). Mirrors the ``_post_merge_llm_fixups``
           behavior the personal-save flow already performs, so
           ``org.agent_settings.llm`` and every member's
           ``agent_settings_diff.llm`` carry a usable, self-describing
           base URL.
    """
    # Arrange + Act — covers: OpenHands explicit null, OpenHands missing,
    # BYOR provider explicit null (base_url auto-filled to provider default).
    openhands_null = OrgLLMSettingsUpdate.model_validate(
        {
            "agent_settings": {
                "llm": {"model": "openhands/claude-3", "base_url": None},
            },
        }
    )
    openhands_missing = OrgLLMSettingsUpdate.model_validate(
        {"agent_settings": {"llm": {"model": "openhands/claude-3"}}}
    )
    anthropic_null = OrgLLMSettingsUpdate.model_validate(
        {
            "agent_settings": {
                "llm": {"model": "anthropic/claude-3-opus-20240229", "base_url": None},
            },
        }
    )

    # Assert — OpenHands gets the proxy URL; non-OpenHands provider gets the
    # provider default that ``litellm.get_api_base`` reports.
    openhands_null_patch = openhands_null.agent_settings_patch()
    assert openhands_null_patch is not None
    assert openhands_null_patch["llm"]["model"] == "litellm_proxy/claude-3"
    assert openhands_null_patch["llm"]["base_url"].rstrip(
        "/"
    ) == LITE_LLM_API_URL.rstrip("/")

    openhands_missing_patch = openhands_missing.agent_settings_patch()
    assert openhands_missing_patch is not None
    assert openhands_missing_patch["llm"]["model"] == "litellm_proxy/claude-3"
    assert openhands_missing_patch["llm"]["base_url"].rstrip(
        "/"
    ) == LITE_LLM_API_URL.rstrip("/")

    anthropic_patch = anthropic_null.agent_settings_patch()
    assert anthropic_patch is not None
    anthropic_base = anthropic_patch["llm"]["base_url"]
    assert isinstance(anthropic_base, str)
    assert "anthropic.com" in anthropic_base


def test_from_org_denormalizes_litellm_proxy_prefix_and_returns_base_url_as_stored():
    """GIVEN: An org whose stored ``agent_settings.llm.model`` is in the SDK's
           normalized ``litellm_proxy/`` form with ``base_url`` equal to the
           managed proxy URL (the state produced by
           ``_normalize_agent_settings`` on save)
    WHEN:  OrgLLMSettingsResponse.from_org serializes for the frontend
    THEN:  the response shows ``openhands/X`` so the basic-view provider
           dropdown matches, returns ``base_url`` exactly as stored so the
           three sync targets (``org.agent_settings.llm.base_url``,
           ``org_member.agent_settings_diff.llm.base_url``, and this
           response) agree, and nulls ``api_key`` so neither the raw secret
           nor the ``MASKED_API_KEY`` marker leaks in the response.
    """
    # Arrange
    org = MagicMock(spec=Org)
    org.agent_settings = {
        "schema_version": 1,
        "agent": "CodeActAgent",
        "llm": {
            "model": "litellm_proxy/minimax-m2.5",
            "base_url": LITE_LLM_API_URL,
            "api_key": MASKED_API_KEY,
        },
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None

    # Act
    response = OrgLLMSettingsResponse.from_org(org)

    # Assert
    assert response.agent_settings.llm.model == "openhands/minimax-m2.5"
    assert response.agent_settings.llm.base_url == LITE_LLM_API_URL
    assert response.agent_settings.llm.api_key is None


def test_from_org_returns_provider_default_base_url_as_stored_for_non_managed_models():
    """GIVEN: An org saved with a BYOR provider whose stored ``base_url`` equals
           that provider's canonical base URL (what ``_normalize_agent_settings``
           auto-filled on save via ``get_provider_api_base``)
    WHEN:  OrgLLMSettingsResponse.from_org serializes for the frontend
    THEN:  ``base_url`` is passed through unchanged — the response must
           reflect stored state so a future drift that re-introduces
           clearing fails this test. The frontend
           (``KNOWN_PROVIDER_DEFAULT_BASE_URLS``) is responsible for
           recognizing provider defaults as "basic mode."
    """
    # Arrange — look up the canonical anthropic base URL the same way the
    # validator does so the test stays in sync with whatever litellm reports.
    from openhands.utils.llm import get_provider_api_base as _provider_base

    anthropic_default = _provider_base("anthropic/claude-3-opus-20240229")
    assert anthropic_default is not None

    org = MagicMock(spec=Org)
    org.agent_settings = {
        "schema_version": 1,
        "agent": "CodeActAgent",
        "llm": {
            "model": "anthropic/claude-3-opus-20240229",
            "base_url": anthropic_default,
        },
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None

    # Act
    response = OrgLLMSettingsResponse.from_org(org)

    # Assert — model is unchanged (no litellm_proxy/ prefix), base_url is
    # returned as stored so stored state and response agree.
    assert response.agent_settings.llm.model == "anthropic/claude-3-opus-20240229"
    assert response.agent_settings.llm.base_url == anthropic_default


def test_from_org_keeps_custom_base_url_that_is_not_provider_default():
    """GIVEN: An org saved with a BYOR provider and a genuinely custom base URL
           (e.g. a company-run proxy) that does NOT match the provider default
    WHEN:  OrgLLMSettingsResponse.from_org serializes for the frontend
    THEN:  ``base_url`` is preserved so the "advanced" view can display it —
           we only clear values we're certain match a canonical default.
    """
    # Arrange
    org = MagicMock(spec=Org)
    org.agent_settings = {
        "schema_version": 1,
        "agent": "CodeActAgent",
        "llm": {
            "model": "anthropic/claude-3-opus-20240229",
            "base_url": "https://company-proxy.internal/anthropic",
        },
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None

    # Act
    response = OrgLLMSettingsResponse.from_org(org)

    # Assert
    assert (
        response.agent_settings.llm.base_url
        == "https://company-proxy.internal/anthropic"
    )
