"""
Unit tests for OrgLLMSettingsService.

Tests the service layer for organization LLM settings operations.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from server.constants import LITE_LLM_API_URL
from server.routes.org_models import (
    MASKED_API_KEY,
    OrgLLMSettingsResponse,
    OrgLLMSettingsUpdate,
    OrgNotFoundError,
)
from server.services import org_llm_settings_service as service_module
from server.services.org_llm_settings_service import OrgLLMSettingsService
from storage.org import Org
from storage.org_member import OrgMember
from storage.org_member_store import OrgMemberStore


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
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'claude-3',
            'base_url': 'https://api.anthropic.com',
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
    """
    GIVEN: A user with a current organization
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
    assert result.agent_settings.llm.model == 'claude-3'
    assert result.agent_settings.agent == 'CodeActAgent'
    mock_store.get_current_org_by_user_id.assert_called_once_with(user_id)


@pytest.mark.asyncio
async def test_get_org_llm_settings_user_not_authenticated(mock_store):
    """
    GIVEN: A user is not authenticated
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

    assert 'not authenticated' in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_org_llm_settings_org_not_found(
    user_id, mock_store, mock_user_context
):
    """
    GIVEN: A user has no current organization
    WHEN: get_org_llm_settings is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=None)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(OrgNotFoundError) as exc_info:
        await service.get_org_llm_settings()

    assert 'No current organization' in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_org_llm_settings_success(
    user_id, mock_org, mock_store, mock_user_context, monkeypatch
):
    """
    GIVEN: A user with a current organization
    WHEN: update_org_llm_settings is called with new values
    THEN: OrgLLMSettingsResponse is returned with updated data
    """
    # Arrange
    updated_org = MagicMock(spec=Org)
    updated_org.id = mock_org.id
    updated_org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {'model': 'new-model'},
    }
    updated_org.conversation_settings = {
        'confirmation_mode': False,
        'max_iterations': 100,
    }
    updated_org.llm_api_key = None
    updated_org.search_api_key = None

    update_data = OrgLLMSettingsUpdate(
        agent_settings={
            'llm': {'model': 'new-model'},
        },
        conversation_settings={
            'confirmation_mode': False,
            'max_iterations': 100,
        },
    )

    mock_store.db_session = MagicMock()
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_llm_settings = AsyncMock(return_value=updated_org)
    monkeypatch.setattr(
        OrgMemberStore, 'update_all_members_llm_settings_async', AsyncMock()
    )
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.update_org_llm_settings(update_data)

    # Assert
    assert isinstance(result, OrgLLMSettingsResponse)
    assert result.agent_settings.llm.model == 'new-model'
    assert result.conversation_settings.confirmation_mode is False
    assert result.conversation_settings.max_iterations == 100
    mock_store.update_org_llm_settings.assert_called_once_with(
        org_id=mock_org.id,
        update_data=update_data,
    )


@pytest.mark.asyncio
async def test_update_org_llm_settings_no_changes(
    user_id, mock_org, mock_store, mock_user_context
):
    """
    GIVEN: A user with a current organization
    WHEN: update_org_llm_settings is called with no fields
    THEN: Current settings are returned without calling update
    """
    # Arrange
    update_data = OrgLLMSettingsUpdate()  # No fields set

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_llm_settings = AsyncMock()
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.update_org_llm_settings(update_data)

    # Assert
    assert isinstance(result, OrgLLMSettingsResponse)
    assert result.agent_settings.llm.model == 'claude-3'
    mock_store.update_org_llm_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_org_llm_settings_org_not_found(
    user_id, mock_store, mock_user_context
):
    """
    GIVEN: A user has no current organization
    WHEN: update_org_llm_settings is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    update_data = OrgLLMSettingsUpdate(agent_settings={'llm': {'model': 'new-model'}})

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=None)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(OrgNotFoundError) as exc_info:
        await service.update_org_llm_settings(update_data)

    assert 'No current organization' in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_org_llm_settings_accepts_wire_payload_and_propagates(
    mock_org, mock_store, mock_user_context, monkeypatch
):
    """
    GIVEN: The exact wire payload the frontend posts to /api/organizations/llm
           ({"agent_settings": {...}}) and an org with members carrying stale
           agent_settings_diff overrides
    WHEN:  update_org_llm_settings is called
    THEN:  1) the org is actually updated (was a silent no-op before the
             field-name fix because the payload keys didn't match the model),
           2) OrgMemberStore.update_all_members_llm_settings_async is invoked
             so members' stale diffs are overwritten by the new org defaults
    """
    # Arrange
    updated_org = MagicMock(spec=Org)
    updated_org.id = mock_org.id
    updated_org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'claude-3',
            'base_url': 'https://llm-proxy.staging.all-hands.dev',
        },
    }
    updated_org.conversation_settings = {}
    updated_org.llm_api_key = None
    updated_org.search_api_key = None

    update_data = OrgLLMSettingsUpdate.model_validate(
        {
            'agent_settings': {
                'llm': {'base_url': 'https://llm-proxy.staging.all-hands.dev'},
            },
        }
    )

    mock_store.db_session = MagicMock()
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_llm_settings = AsyncMock(return_value=updated_org)

    propagation_mock = AsyncMock()
    monkeypatch.setattr(
        OrgMemberStore,
        'update_all_members_llm_settings_async',
        propagation_mock,
    )

    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    await service.update_org_llm_settings(update_data)

    # Assert
    mock_store.update_org_llm_settings.assert_awaited_once_with(
        org_id=mock_org.id,
        update_data=update_data,
    )
    propagation_mock.assert_awaited_once()
    call_args = propagation_mock.await_args
    assert call_args.args[0] is mock_store.db_session
    assert call_args.args[1] == mock_org.id
    member_settings = call_args.args[2]
    assert member_settings.agent_settings_diff == {
        'llm': {'base_url': 'https://llm-proxy.staging.all-hands.dev'},
    }
    assert member_settings.conversation_settings_diff is None
    assert member_settings.llm_api_key is None


@pytest.mark.asyncio
async def test_update_org_llm_settings_does_not_propagate_when_only_org_fields(
    mock_org, mock_store, mock_user_context, monkeypatch
):
    """
    GIVEN: An update that only touches org-scoped fields (search_api_key),
           with nothing that should land on members
    WHEN:  update_org_llm_settings is called
    THEN:  member propagation is not invoked, guarding against regressions
           that would needlessly rewrite every member's row
    """
    # Arrange
    updated_org = MagicMock(spec=Org)
    updated_org.id = mock_org.id
    updated_org.agent_settings = mock_org.agent_settings
    updated_org.conversation_settings = {}
    updated_org.llm_api_key = None
    updated_org.search_api_key = None

    update_data = OrgLLMSettingsUpdate(search_api_key='new-search-key')

    mock_store.db_session = MagicMock()
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_llm_settings = AsyncMock(return_value=updated_org)

    propagation_mock = AsyncMock()
    monkeypatch.setattr(
        OrgMemberStore,
        'update_all_members_llm_settings_async',
        propagation_mock,
    )

    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    await service.update_org_llm_settings(update_data)

    # Assert
    propagation_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_org_llm_settings_lifts_nested_api_key_for_column_sync(
    mock_org, mock_store, mock_user_context, monkeypatch
):
    """
    GIVEN: The frontend posts the api_key nested inside agent_settings.llm
    WHEN:  update_org_llm_settings is called
    THEN:  the raw key is routed through the top-level llm_api_key field,
           so every member's encrypted _llm_api_key column is updated via
           the propagation call. Without this, members' encrypted columns
           stay stale and _get_effective_llm_api_key returns the old key
           when they refetch settings.
    """
    # Arrange
    updated_org = MagicMock(spec=Org)
    updated_org.id = mock_org.id
    updated_org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {'model': 'claude-3', 'base_url': 'https://example.com'},
    }
    updated_org.conversation_settings = {}
    updated_org.llm_api_key = None
    updated_org.search_api_key = None

    update_data = OrgLLMSettingsUpdate.model_validate(
        {
            'agent_settings': {
                'llm': {
                    'api_key': 'sk-new-raw-key',
                    'base_url': 'https://example.com',
                },
            },
        }
    )

    # The validator must move the raw key out of agent_settings and into the
    # top-level field the rest of the pipeline actually wires to the column,
    # AND leave a masked marker in the JSON so ``org.agent_settings.llm``
    # and member ``agent_settings_diff.llm`` stay consistent with the
    # encrypted column.
    assert update_data.llm_api_key == 'sk-new-raw-key'
    assert update_data.agent_settings == {
        'llm': {'api_key': MASKED_API_KEY, 'base_url': 'https://example.com'},
    }

    mock_store.db_session = MagicMock()
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_llm_settings = AsyncMock(return_value=updated_org)

    propagation_mock = AsyncMock()
    monkeypatch.setattr(
        OrgMemberStore,
        'update_all_members_llm_settings_async',
        propagation_mock,
    )

    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    await service.update_org_llm_settings(update_data)

    # Assert
    propagation_mock.assert_awaited_once()
    member_settings = propagation_mock.await_args.args[2]
    assert isinstance(member_settings.llm_api_key, SecretStr)
    assert member_settings.llm_api_key.get_secret_value() == 'sk-new-raw-key'
    assert member_settings.agent_settings_diff == {
        'llm': {'api_key': MASKED_API_KEY, 'base_url': 'https://example.com'},
    }
    # org-defaults saves are org-wide: always reset members' personal
    # BYOR flag so load-time fallthrough won't return a stale custom key.
    assert member_settings.has_custom_llm_api_key is False


def test_get_member_updates_treats_empty_llm_api_key_as_none():
    """
    GIVEN: The frontend posts an empty api_key to signal "switch to the
           managed/OpenHands provider" (clear the org-wide custom key)
    WHEN:  get_member_updates builds the payload for member propagation
    THEN:  llm_api_key is coerced to None — because OrgMember.llm_api_key's
           setter has no ``if raw else None`` guard (the column is
           ``nullable=False``), an empty string would be encrypted into an
           empty blob on every member rather than cleared. agent_settings_diff
           still carries the model diff so the org default actually lands on
           members.
    """
    # Arrange
    update = OrgLLMSettingsUpdate(
        agent_settings={'llm': {'model': 'openhands/x'}}, llm_api_key=''
    )

    # Act
    member_updates = update.get_member_updates()

    # Assert — llm_api_key is coerced to None, and the agent_settings_diff
    # carries both the model diff AND the auto-filled LiteLLM proxy URL so
    # the stored state is self-describing.
    assert member_updates is not None
    assert member_updates.llm_api_key is None
    assert member_updates.agent_settings_diff == {
        'llm': {'model': 'openhands/x', 'base_url': LITE_LLM_API_URL},
    }


@pytest.mark.asyncio
async def test_update_org_llm_settings_generates_managed_key_for_openhands(
    user_id, mock_org, mock_store, mock_user_context, monkeypatch
):
    """
    GIVEN: An admin or owner saves org-defaults with the OpenHands provider
           in basic view (payload: model="openhands/...", api_key="",
           base_url=None), and their OrgMember row still holds the custom key
           from a prior *All*-tab save
    WHEN:  update_org_llm_settings runs
    THEN:  a managed LiteLLM key is generated for the acting user, their
           has_custom_llm_api_key flag is cleared, and a single propagation
           call writes the generated key + agent_settings_diff to every
           member row in one DB pass — so the whole org shares one managed
           key column value without doubling the row-write churn.
    """
    # Arrange
    updated_org = MagicMock(spec=Org)
    updated_org.id = mock_org.id
    updated_org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {'model': 'openhands/claude-3'},
    }
    updated_org.conversation_settings = {}
    updated_org.llm_api_key = None
    updated_org.search_api_key = None

    update_data = OrgLLMSettingsUpdate.model_validate(
        {
            'agent_settings': {
                'llm': {
                    'model': 'openhands/claude-3',
                    'api_key': '',
                    'base_url': None,
                },
            },
        }
    )

    # The acting user's member row — carries the stale custom key from the
    # prior *All*-tab save.
    acting_member = MagicMock(spec=OrgMember)
    acting_member.llm_api_key = SecretStr('old-custom-key')
    acting_member.has_custom_llm_api_key = True

    scalars_result = MagicMock()
    scalars_result.first.return_value = acting_member
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars_result

    db_session = MagicMock()
    db_session.execute = AsyncMock(return_value=execute_result)

    mock_store.db_session = db_session
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_llm_settings = AsyncMock(return_value=updated_org)

    propagation_mock = AsyncMock()
    monkeypatch.setattr(
        OrgMemberStore,
        'update_all_members_llm_settings_async',
        propagation_mock,
    )
    monkeypatch.setattr(
        service_module.LiteLlmManager,
        'verify_existing_key',
        AsyncMock(return_value=False),
    )
    generate_key_mock = AsyncMock(return_value='litellm-new-key')
    monkeypatch.setattr(
        service_module.LiteLlmManager, 'generate_key', generate_key_mock
    )

    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    await service.update_org_llm_settings(update_data)

    # Assert — key was generated with openhands metadata.
    generate_key_mock.assert_awaited_once()
    assert generate_key_mock.await_args.args[0] == user_id
    assert generate_key_mock.await_args.args[1] == str(mock_org.id)
    assert generate_key_mock.await_args.args[2] is None
    assert generate_key_mock.await_args.args[3] == {'type': 'openhands'}

    # Assert — a single propagation call writes the diff, the generated
    # key, AND the has_custom_llm_api_key=False reset to every member row
    # (including the acting user's own row) in one DB pass.
    propagation_mock.assert_awaited_once()
    member_settings = propagation_mock.await_args.args[2]
    assert isinstance(member_settings.llm_api_key, SecretStr)
    assert member_settings.llm_api_key.get_secret_value() == 'litellm-new-key'
    assert member_settings.has_custom_llm_api_key is False
    assert member_settings.agent_settings_diff == {
        'llm': {
            'model': 'openhands/claude-3',
            'api_key': MASKED_API_KEY,
            'base_url': LITE_LLM_API_URL,
        },
    }


@pytest.mark.asyncio
async def test_update_org_llm_settings_does_not_generate_key_for_non_managed_model(
    mock_org, mock_store, mock_user_context, monkeypatch
):
    """
    GIVEN: The owner saves org-defaults with a BYOR model (non-OpenHands,
           non-proxy base_url)
    WHEN:  update_org_llm_settings runs
    THEN:  no LiteLLM key is generated — the managed-key path is reserved for
           the OpenHands provider / managed-proxy case, mirroring the personal
           save flow's detection logic.
    """
    # Arrange
    updated_org = MagicMock(spec=Org)
    updated_org.id = mock_org.id
    updated_org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'anthropic/claude-3',
            'base_url': 'https://example.com',
        },
    }
    updated_org.conversation_settings = {}
    updated_org.llm_api_key = None
    updated_org.search_api_key = None

    update_data = OrgLLMSettingsUpdate.model_validate(
        {
            'agent_settings': {
                'llm': {
                    'model': 'anthropic/claude-3',
                    'base_url': 'https://example.com',
                    'api_key': 'custom',
                },
            },
        }
    )

    mock_store.db_session = MagicMock()
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_llm_settings = AsyncMock(return_value=updated_org)

    monkeypatch.setattr(
        OrgMemberStore,
        'update_all_members_llm_settings_async',
        AsyncMock(),
    )
    generate_key_mock = AsyncMock(return_value='should-not-be-called')
    monkeypatch.setattr(
        service_module.LiteLlmManager, 'generate_key', generate_key_mock
    )
    monkeypatch.setattr(
        service_module.LiteLlmManager,
        'verify_existing_key',
        AsyncMock(return_value=True),
    )

    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    await service.update_org_llm_settings(update_data)

    # Assert
    generate_key_mock.assert_not_awaited()


def test_normalize_agent_settings_masks_api_key_in_json_on_empty_and_real_keys():
    """
    GIVEN: Wire payloads that either carry a real raw api_key (BYOR save) or
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
        {'agent_settings': {'llm': {'model': 'anthropic/x', 'api_key': 'sk-raw'}}}
    )
    empty_key = OrgLLMSettingsUpdate.model_validate(
        {
            'agent_settings': {
                'llm': {'model': 'openhands/x', 'api_key': '', 'base_url': None},
            },
        }
    )

    # Assert — masked in JSON in both cases; lifted raw value on top-level.
    assert real_key.llm_api_key == 'sk-raw'
    assert real_key.agent_settings is not None
    assert real_key.agent_settings['llm']['api_key'] == MASKED_API_KEY
    assert empty_key.llm_api_key == ''
    assert empty_key.agent_settings is not None
    assert empty_key.agent_settings['llm']['api_key'] == MASKED_API_KEY


def test_normalize_agent_settings_fills_base_url_for_all_providers():
    """
    GIVEN: Wire payloads from the basic view that send ``base_url: null`` for
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
            'agent_settings': {
                'llm': {'model': 'openhands/claude-3', 'base_url': None},
            },
        }
    )
    openhands_missing = OrgLLMSettingsUpdate.model_validate(
        {'agent_settings': {'llm': {'model': 'openhands/claude-3'}}}
    )
    anthropic_null = OrgLLMSettingsUpdate.model_validate(
        {
            'agent_settings': {
                'llm': {'model': 'anthropic/claude-3-opus-20240229', 'base_url': None},
            },
        }
    )

    # Assert — OpenHands gets the proxy URL; non-OpenHands provider gets the
    # provider default that ``litellm.get_api_base`` reports.
    assert openhands_null.agent_settings is not None
    assert openhands_null.agent_settings['llm']['base_url'] == LITE_LLM_API_URL
    assert openhands_missing.agent_settings is not None
    assert openhands_missing.agent_settings['llm']['base_url'] == LITE_LLM_API_URL
    assert anthropic_null.agent_settings is not None
    # get_provider_api_base('anthropic/claude-3-opus-20240229') returns the
    # Anthropic public API. Be lenient about the exact suffix litellm returns
    # across versions — only require that it got filled with Anthropic's host.
    anthropic_base = anthropic_null.agent_settings['llm']['base_url']
    assert isinstance(anthropic_base, str)
    assert 'anthropic.com' in anthropic_base


def test_from_org_denormalizes_litellm_proxy_prefix_and_returns_base_url_as_stored():
    """
    GIVEN: An org whose stored ``agent_settings.llm.model`` is in the SDK's
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

    # Act
    response = OrgLLMSettingsResponse.from_org(org)

    # Assert
    assert response.agent_settings.llm.model == 'openhands/minimax-m2.5'
    assert response.agent_settings.llm.base_url == LITE_LLM_API_URL
    assert response.agent_settings.llm.api_key is None


def test_from_org_returns_provider_default_base_url_as_stored_for_non_managed_models():
    """
    GIVEN: An org saved with a BYOR provider whose stored ``base_url`` equals
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

    # Act
    response = OrgLLMSettingsResponse.from_org(org)

    # Assert — model is unchanged (no litellm_proxy/ prefix), base_url is
    # returned as stored so stored state and response agree.
    assert response.agent_settings.llm.model == 'anthropic/claude-3-opus-20240229'
    assert response.agent_settings.llm.base_url == anthropic_default


def test_from_org_keeps_custom_base_url_that_is_not_provider_default():
    """
    GIVEN: An org saved with a BYOR provider and a genuinely custom base URL
           (e.g. a company-run proxy) that does NOT match the provider default
    WHEN:  OrgLLMSettingsResponse.from_org serializes for the frontend
    THEN:  ``base_url`` is preserved so the "advanced" view can display it —
           we only clear values we're certain match a canonical default.
    """
    # Arrange
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
    org.search_api_key = None

    # Act
    response = OrgLLMSettingsResponse.from_org(org)

    # Assert
    assert (
        response.agent_settings.llm.base_url
        == 'https://company-proxy.internal/anthropic'
    )
