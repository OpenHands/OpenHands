from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from integrations.jira_dc.jira_dc_conversation_secret_enricher import (
    JiraDcConversationSecretEnricher,
)
from integrations.jira_dc.jira_dc_user_token import (
    JiraDcUserToken,
    JiraDcUserTokenError,
)
from pydantic import SecretStr

from openhands.app_server.app_conversation.app_conversation_models import (
    ConversationTrigger,
)
from openhands.sdk.secret import LookupSecret, StaticSecret

ORG_ID = UUID('00000000-0000-0000-0000-000000000123')
OTHER_ORG_ID = UUID('00000000-0000-0000-0000-000000000456')


class FakeUserContext:
    def __init__(self, *, user_id: str = 'kc-user', org_id: UUID | None = ORG_ID):
        self.user_id = user_id
        self.user_auth = MagicMock()
        self.user_auth.get_effective_org_id = AsyncMock(return_value=org_id)

    async def get_user_id(self) -> str:
        return self.user_id


def _linked_store(*, org_id: UUID | None = ORG_ID, admin_user_id: str = 'admin-user'):
    store = MagicMock()
    store.get_user_by_active_workspace = AsyncMock(
        return_value=SimpleNamespace(jira_dc_workspace_id=7)
    )
    store.get_workspace_by_id = AsyncMock(
        return_value=SimpleNamespace(
            id=7,
            name='jira.example.com',
            status='active',
            org_id=org_id,
            admin_user_id=admin_user_id,
        )
    )
    return store


@pytest.mark.asyncio
async def test_enricher_adds_jira_dc_lookup_secret_for_linked_user():
    store = _linked_store()
    jwt_service = MagicMock()
    jwt_service.create_jws_token.return_value = 'signed-token'

    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            True,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_BASE_URL',
            'https://jira.example.com',
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
    ):
        enrichment = await JiraDcConversationSecretEnricher().enrich(
            user_context=FakeUserContext(),
            user=MagicMock(id='kc-user'),
            trigger=ConversationTrigger.SLACK,
            system_message_suffix='Existing instructions.',
            web_url='https://openhands.example.com',
            jwt_service=jwt_service,
            access_token_hard_timeout=timedelta(minutes=5),
        )

    assert isinstance(enrichment.secrets['JIRA_DC_BASE_URL'], StaticSecret)
    assert (
        enrichment.secrets['JIRA_DC_BASE_URL'].value.get_secret_value()
        == 'https://jira.example.com'
    )
    token_secret = enrichment.secrets['JIRA_DC_TOKEN']
    assert isinstance(token_secret, LookupSecret)
    assert (
        token_secret.url
        == 'https://openhands.example.com/integration/jira-dc/secrets/token'
    )
    assert token_secret.headers == {'X-Access-Token': 'signed-token'}
    assert 'JIRA_DC_TOKEN' in (enrichment.system_message_suffix or '')


@pytest.mark.asyncio
async def test_enricher_skips_org_scoped_workspace_when_context_org_differs():
    store = _linked_store(org_id=OTHER_ORG_ID)

    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            True,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_BASE_URL',
            'https://jira.example.com',
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
    ):
        enrichment = await JiraDcConversationSecretEnricher().enrich(
            user_context=FakeUserContext(org_id=ORG_ID),
            user=MagicMock(id='kc-user'),
            trigger=ConversationTrigger.SLACK,
            system_message_suffix='Existing instructions.',
            web_url='https://openhands.example.com',
            jwt_service=MagicMock(),
            access_token_hard_timeout=timedelta(minutes=5),
        )

    assert enrichment.secrets == {}
    assert enrichment.system_message_suffix == 'Existing instructions.'


@pytest.mark.asyncio
async def test_enricher_validates_token_before_jira_triggered_start():
    store = _linked_store()
    get_token = AsyncMock(
        return_value=JiraDcUserToken(access_token=SecretStr('token'), expires_at=0)
    )
    jwt_service = MagicMock()
    jwt_service.create_jws_token.return_value = 'signed-token'

    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            True,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_BASE_URL',
            'https://jira.example.com',
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.get_user_jira_dc_token',
            get_token,
        ),
    ):
        await JiraDcConversationSecretEnricher().enrich(
            user_context=FakeUserContext(),
            user=MagicMock(id='kc-user'),
            trigger=ConversationTrigger.JIRA,
            system_message_suffix=None,
            web_url='https://openhands.example.com',
            jwt_service=jwt_service,
            access_token_hard_timeout=timedelta(minutes=5),
        )

    get_token.assert_awaited_once()


@pytest.mark.asyncio
async def test_enricher_propagates_token_error_for_jira_triggered_start():
    store = _linked_store()

    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            True,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_BASE_URL',
            'https://jira.example.com',
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.get_user_jira_dc_token',
            AsyncMock(side_effect=JiraDcUserTokenError('re-link required')),
        ),
    ):
        with pytest.raises(JiraDcUserTokenError):
            await JiraDcConversationSecretEnricher().enrich(
                user_context=FakeUserContext(),
                user=MagicMock(id='kc-user'),
                trigger=ConversationTrigger.JIRA,
                system_message_suffix=None,
                web_url='https://openhands.example.com',
                jwt_service=MagicMock(),
                access_token_hard_timeout=timedelta(minutes=5),
            )


# A personal org's id == its owner's user id, so a JDC workspace created in
# personal-workspace mode is stamped org_id == admin_user_id.
PERSONAL_ORG_ID = UUID('00000000-0000-0000-0000-000000000789')


@pytest.mark.asyncio
async def test_enricher_allows_personal_workspace_org_for_non_creator():
    # Personal-workspace install: the JDC workspace (one company's Jira) is
    # stamped with its creator's personal org. A *different* user must still get
    # their own token -- it's an instance-wide integration, not a tenant boundary.
    store = _linked_store(org_id=PERSONAL_ORG_ID, admin_user_id=str(PERSONAL_ORG_ID))
    jwt_service = MagicMock()
    jwt_service.create_jws_token.return_value = 'signed-token'

    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            True,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_BASE_URL',
            'https://jira.example.com',
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
    ):
        enrichment = await JiraDcConversationSecretEnricher().enrich(
            user_context=FakeUserContext(user_id='other-user', org_id=ORG_ID),
            user=MagicMock(id='other-user'),
            trigger=ConversationTrigger.SLACK,
            system_message_suffix=None,
            web_url='https://openhands.example.com',
            jwt_service=jwt_service,
            access_token_hard_timeout=timedelta(minutes=5),
        )

    assert 'JIRA_DC_TOKEN' in enrichment.secrets
    assert 'JIRA_DC_TOKEN' in (enrichment.system_message_suffix or '')


@pytest.mark.asyncio
async def test_enricher_still_blocks_real_cross_org_workspace():
    # A genuine team/default org (org_id != admin_user_id) keeps enforcing
    # isolation: a user from a different org gets no token.
    store = _linked_store(org_id=OTHER_ORG_ID, admin_user_id='admin-user')

    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            True,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_BASE_URL',
            'https://jira.example.com',
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
    ):
        enrichment = await JiraDcConversationSecretEnricher().enrich(
            user_context=FakeUserContext(org_id=ORG_ID),
            user=MagicMock(id='kc-user'),
            trigger=ConversationTrigger.SLACK,
            system_message_suffix='Existing instructions.',
            web_url='https://openhands.example.com',
            jwt_service=MagicMock(),
            access_token_hard_timeout=timedelta(minutes=5),
        )

    assert enrichment.secrets == {}
    assert enrichment.system_message_suffix == 'Existing instructions.'


@pytest.mark.asyncio
async def test_enricher_injects_for_null_org_workspace():
    # An unscoped workspace (org_id is None) is instance-wide -- regression guard.
    store = _linked_store(org_id=None)
    jwt_service = MagicMock()
    jwt_service.create_jws_token.return_value = 'signed-token'

    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            True,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_BASE_URL',
            'https://jira.example.com',
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
    ):
        enrichment = await JiraDcConversationSecretEnricher().enrich(
            user_context=FakeUserContext(user_id='anyone', org_id=None),
            user=MagicMock(id='anyone'),
            trigger=ConversationTrigger.SLACK,
            system_message_suffix=None,
            web_url='https://openhands.example.com',
            jwt_service=jwt_service,
            access_token_hard_timeout=timedelta(minutes=5),
        )

    assert 'JIRA_DC_TOKEN' in enrichment.secrets


@pytest.mark.asyncio
async def test_enricher_email_mode_hint_when_configured():
    # Email mode: no token, but a set-up connection => tell the agent how to
    # point the user at enabling live access (OAuth / Secret / MCP).
    store = MagicMock()
    store.get_active_workspace = AsyncMock(
        return_value=SimpleNamespace(
            id=1, name='jira.example.com', status='active', org_id=None
        )
    )
    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            False,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
    ):
        enrichment = await JiraDcConversationSecretEnricher().enrich(
            user_context=FakeUserContext(),
            user=MagicMock(id='kc-user'),
            trigger=ConversationTrigger.SLACK,
            system_message_suffix='Existing instructions.',
            web_url=None,
            jwt_service=MagicMock(),
            access_token_hard_timeout=timedelta(minutes=5),
        )

    assert enrichment.secrets == {}
    suffix = enrichment.system_message_suffix or ''
    assert 'Existing instructions.' in suffix
    # Host comes from the active workspace, not the (email-mode-empty) env var.
    assert 'jira.example.com' in suffix
    # Hedged: use a token/MCP tools if available, otherwise list the options.
    assert 'available Jira token' in suffix
    assert 'Jira MCP tools' in suffix
    assert 'OAuth mode' in suffix
    assert 'as a Secret' in suffix
    assert 'MCP server' in suffix


@pytest.mark.asyncio
async def test_enricher_email_mode_no_hint_for_other_org():
    # Active workspace belongs to a different team org => not visible here.
    store = MagicMock()
    store.get_active_workspace = AsyncMock(
        return_value=SimpleNamespace(
            id=1,
            name='jira.example.com',
            status='active',
            org_id=OTHER_ORG_ID,
            admin_user_id='some-other-admin',
        )
    )
    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            False,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
    ):
        enrichment = await JiraDcConversationSecretEnricher().enrich(
            user_context=FakeUserContext(org_id=ORG_ID),
            user=MagicMock(id='kc-user'),
            trigger=ConversationTrigger.SLACK,
            system_message_suffix='Existing instructions.',
            web_url=None,
            jwt_service=MagicMock(),
            access_token_hard_timeout=timedelta(minutes=5),
        )

    assert enrichment.secrets == {}
    assert enrichment.system_message_suffix == 'Existing instructions.'


@pytest.mark.asyncio
async def test_enricher_email_mode_no_hint_when_not_configured():
    # Email mode with no active workspace => nothing set up, so no hint.
    store = MagicMock()
    store.get_active_workspace = AsyncMock(return_value=None)
    with (
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JIRA_DC_ENABLE_OAUTH',
            False,
        ),
        patch(
            'integrations.jira_dc.jira_dc_conversation_secret_enricher.JiraDcIntegrationStore.get_instance',
            return_value=store,
        ),
    ):
        enrichment = await JiraDcConversationSecretEnricher().enrich(
            user_context=FakeUserContext(),
            user=MagicMock(id='kc-user'),
            trigger=ConversationTrigger.SLACK,
            system_message_suffix='Existing instructions.',
            web_url=None,
            jwt_service=MagicMock(),
            access_token_hard_timeout=timedelta(minutes=5),
        )

    assert enrichment.secrets == {}
    assert enrichment.system_message_suffix == 'Existing instructions.'
