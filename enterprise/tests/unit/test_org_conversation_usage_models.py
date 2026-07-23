"""Query-level tests for the model-usage aggregation in org usage stats."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from server.services.org_conversation_service import OrgConversationService
from storage.stored_conversation_cost_event import StoredConversationCostEvent
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas

ORG_ID = uuid4()
USER_ID = uuid4()


def _conversation(session, conversation_id, llm_model, cost, prompt, completion):
    session.add(
        StoredConversationMetadata(
            conversation_id=conversation_id,
            conversation_version='V1',
            llm_model=llm_model,
            accumulated_cost=cost,
            prompt_tokens=prompt,
            completion_tokens=completion,
            created_at=datetime.now(UTC) - timedelta(days=1),
        )
    )
    session.add(
        StoredConversationMetadataSaas(
            conversation_id=conversation_id,
            user_id=USER_ID,
            org_id=ORG_ID,
        )
    )


@pytest.mark.asyncio
async def test_model_usage_ledger_legacy_and_no_event_rows(async_session_maker):
    occurred = datetime.now(UTC) - timedelta(hours=2)
    async with async_session_maker() as session:
        # A: attributed ledger rows across two models; the conversation's own
        # llm_model label must NOT override per-event attribution.
        _conversation(session, 'conv-a', 'litellm_proxy/current-label', 0.25, 300, 30)
        session.add(
            StoredConversationCostEvent(
                conversation_id='conv-a',
                cost_delta=0.08,
                occurred_at=occurred,
                usage_id='agent',
                llm_model='litellm_proxy/gpt-5.5',
                prompt_tokens=100,
                completion_tokens=10,
            )
        )
        session.add(
            StoredConversationCostEvent(
                conversation_id='conv-a',
                cost_delta=0.17,
                occurred_at=occurred,
                usage_id='profile:opus:x1',
                llm_model='litellm_proxy/claude-opus-4-8',
                prompt_tokens=200,
                completion_tokens=20,
            )
        )
        # B: pre-migration NULL rows fall back to the conversation label;
        # NULL token fields contribute zero tokens.
        _conversation(session, 'conv-b', 'legacy-model', 0.30, 999, 99)
        session.add(
            StoredConversationCostEvent(
                conversation_id='conv-b',
                cost_delta=0.30,
                occurred_at=occurred,
            )
        )
        # C: no ledger rows at all — kept via the legacy aggregation.
        _conversation(session, 'conv-c', 'old-model', 0.55, 50, 5)
        await session.commit()

    async with async_session_maker() as session:
        service = OrgConversationService(db_session=session)
        base_filter = [
            StoredConversationMetadata.conversation_version == 'V1',
            StoredConversationMetadataSaas.org_id == ORG_ID,
        ]
        cutoff = datetime.now(UTC) - timedelta(days=30)
        model_usage = await service._get_model_usage(base_filter, cutoff)

    by_model = {m.model_name: m for m in model_usage}
    assert by_model['litellm_proxy/gpt-5.5'].total_cost == pytest.approx(0.08)
    assert by_model['litellm_proxy/gpt-5.5'].total_tokens == 110
    assert by_model['litellm_proxy/claude-opus-4-8'].total_cost == pytest.approx(0.17)
    assert by_model['litellm_proxy/claude-opus-4-8'].total_tokens == 220
    assert by_model['legacy-model'].total_cost == pytest.approx(0.30)
    assert by_model['legacy-model'].total_tokens == 0
    assert by_model['old-model'].total_cost == pytest.approx(0.55)
    assert by_model['old-model'].total_tokens == 55
    # The relabel-prone conversation label never appears as its own row.
    assert 'litellm_proxy/current-label' not in by_model
    # Ordered by spend, descending.
    costs = [m.total_cost for m in model_usage]
    assert costs == sorted(costs, reverse=True)


async def _seed_three_conversation_scenario(async_session_maker):
    occurred = datetime.now(UTC) - timedelta(hours=2)
    async with async_session_maker() as session:
        _conversation(session, 'conv-a', 'litellm_proxy/current-label', 0.25, 300, 30)
        session.add(
            StoredConversationCostEvent(
                conversation_id='conv-a',
                cost_delta=0.08,
                occurred_at=occurred,
                usage_id='agent',
                llm_model='litellm_proxy/gpt-5.5',
                prompt_tokens=100,
                completion_tokens=10,
            )
        )
        session.add(
            StoredConversationCostEvent(
                conversation_id='conv-a',
                cost_delta=0.17,
                occurred_at=occurred,
                usage_id='profile:opus:x1',
                llm_model='litellm_proxy/claude-opus-4-8',
                prompt_tokens=200,
                completion_tokens=20,
            )
        )
        _conversation(session, 'conv-b', 'legacy-model', 0.30, 999, 99)
        session.add(
            StoredConversationCostEvent(
                conversation_id='conv-b',
                cost_delta=0.30,
                occurred_at=occurred,
            )
        )
        _conversation(session, 'conv-c', 'old-model', 0.55, 50, 5)
        await session.commit()


@pytest.mark.asyncio
async def test_spend_time_totals_team_and_agent_usage(async_session_maker):
    """Headline/team/agent aggregations follow the ledger like the model tab."""
    await _seed_three_conversation_scenario(async_session_maker)

    base_filter = [
        StoredConversationMetadata.conversation_version == 'V1',
        StoredConversationMetadataSaas.org_id == ORG_ID,
    ]
    cutoff = datetime.now(UTC) - timedelta(days=30)

    async with async_session_maker() as session:
        service = OrgConversationService(db_session=session)
        total_cost, prompt, completion = await service._get_usage_totals(
            base_filter, cutoff
        )
        team_tokens = await service._get_team_token_usage(base_filter, cutoff)
        agent_costs = await service._get_agent_cost_usage(base_filter, cutoff)

    # Ledger (0.08 + 0.17 + 0.30 legacy NULL rows) + no-ledger fallback (0.55).
    assert total_cost == pytest.approx(1.10)
    # Pre-attribution NULL rows carry no token data (conv-b contributes 0);
    # no-ledger conversations fall back to their column totals.
    assert prompt == 350
    assert completion == 35

    assert set(team_tokens) == {str(USER_ID)}
    assert team_tokens[str(USER_ID)][0] == 385

    assert set(agent_costs) == {'OpenHands'}
    assert agent_costs['OpenHands'] == pytest.approx(1.10)
