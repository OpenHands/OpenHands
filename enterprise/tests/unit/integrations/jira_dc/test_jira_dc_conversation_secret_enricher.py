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


def _linked_store(*, org_id: UUID | None = ORG_ID):
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
