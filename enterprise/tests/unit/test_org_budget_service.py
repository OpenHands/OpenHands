from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest
from server.constants import ORG_SETTINGS_VERSION
from server.services.org_budget_service import (
    OrgBudgetService,
    _current_cycle_start,
)
from storage.org import Org
from storage.org_budget_settings import OrgBudgetSettings
from storage.org_budget_threshold import OrgBudgetThreshold
from storage.org_user_budget_override import OrgUserBudgetOverride


@pytest.fixture
async def budget_org(async_session_maker):
    org_id = uuid4()
    async with async_session_maker() as session:
        org = Org(
            id=org_id,
            name=f'test-org-{org_id}',
            org_version=ORG_SETTINGS_VERSION,
            enable_proactive_conversation_starters=True,
        )
        session.add(org)
        await session.commit()
    return org


@pytest.mark.asyncio
async def test_roll_cycle_if_needed_updates_cycle(async_session_maker, budget_org):
    async with async_session_maker() as session:
        now = datetime.now(UTC)
        reset_day = 1
        past_cycle_start = _current_cycle_start(now - timedelta(days=40), reset_day)
        settings = OrgBudgetSettings(
            org_id=budget_org.id,
            enabled=True,
            reset_day=reset_day,
            monthly_limit=250.0,
            default_user_monthly_limit=None,
            slack_channel=None,
            slack_team_id=None,
            cycle_start_at=past_cycle_start,
            cycle_start_spend=10.0,
        )
        threshold = OrgBudgetThreshold(
            org_id=budget_org.id,
            percentage=80,
            email_enabled=True,
            slack_enabled=False,
            last_triggered_at=now,
            last_triggered_cycle_start=past_cycle_start,
        )
        session.add(settings)
        session.add(threshold)
        await session.commit()

        service = OrgBudgetService(session)
        overrides: list[OrgUserBudgetOverride] = []

        with (
            patch.object(
                service, '_fetch_team_spend', AsyncMock(return_value=42.5)
            ) as fetch_mock,
            patch.object(service, '_sync_litellm_budgets', AsyncMock()) as sync_mock,
        ):
            rolled = await service._roll_cycle_if_needed(
                settings, [threshold], overrides
            )

        assert rolled is True
        assert settings.cycle_start_at.replace(tzinfo=UTC) == _current_cycle_start(
            now, reset_day
        )
        assert settings.cycle_start_spend == 42.5
        assert threshold.last_triggered_at is None
        assert threshold.last_triggered_cycle_start is None
        fetch_mock.assert_awaited_once_with(settings.org_id)
        sync_mock.assert_awaited_once_with(settings.org_id, settings, overrides)


@pytest.mark.asyncio
async def test_roll_cycle_if_needed_noop(async_session_maker, budget_org):
    async with async_session_maker() as session:
        now = datetime.now(UTC)
        reset_day = 1
        current_cycle_start = _current_cycle_start(now, reset_day)
        settings = OrgBudgetSettings(
            org_id=budget_org.id,
            enabled=True,
            reset_day=reset_day,
            monthly_limit=250.0,
            default_user_monthly_limit=None,
            slack_channel=None,
            slack_team_id=None,
            cycle_start_at=current_cycle_start,
            cycle_start_spend=10.0,
        )
        threshold = OrgBudgetThreshold(
            org_id=budget_org.id,
            percentage=80,
            email_enabled=True,
            slack_enabled=False,
            last_triggered_at=now,
            last_triggered_cycle_start=current_cycle_start,
        )
        session.add(settings)
        session.add(threshold)
        await session.commit()

        service = OrgBudgetService(session)
        with (
            patch.object(service, '_fetch_team_spend', AsyncMock()) as fetch_mock,
            patch.object(service, '_sync_litellm_budgets', AsyncMock()) as sync_mock,
        ):
            rolled = await service._roll_cycle_if_needed(settings, [threshold], [])

        assert rolled is False
        assert settings.cycle_start_at == current_cycle_start
        assert threshold.last_triggered_at == now
        fetch_mock.assert_not_called()
        sync_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_budget_maintenance_syncs_when_cycle_not_rolled(
    async_session_maker, budget_org
):
    async with async_session_maker() as session:
        service = OrgBudgetService(session)
        with patch.object(service, '_sync_litellm_budgets', AsyncMock()) as sync_mock:
            result = await service.run_budget_maintenance(budget_org.id)

    assert result['cycle_rolled'] is False
    sync_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_budget_maintenance_uses_cycle_roll_sync(
    async_session_maker, budget_org
):
    async with async_session_maker() as session:
        reset_day = 1
        settings = OrgBudgetSettings(
            org_id=budget_org.id,
            enabled=True,
            reset_day=reset_day,
            monthly_limit=250.0,
            default_user_monthly_limit=None,
            slack_channel=None,
            slack_team_id=None,
            cycle_start_at=_current_cycle_start(
                datetime.now(UTC) - timedelta(days=40), reset_day
            ),
            cycle_start_spend=10.0,
        )
        session.add(settings)
        await session.commit()

        service = OrgBudgetService(session)
        with (
            patch.object(
                service, '_fetch_team_spend', AsyncMock(return_value=42.5)
            ) as fetch_mock,
            patch.object(service, '_sync_litellm_budgets', AsyncMock()) as sync_mock,
        ):
            result = await service.run_budget_maintenance(budget_org.id)

    assert result['cycle_rolled'] is True
    fetch_mock.assert_awaited_once_with(settings.org_id)
    sync_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_litellm_budgets_updates_team_and_members(
    async_session_maker, budget_org
):
    async with async_session_maker() as session:
        service = OrgBudgetService(session)
        now = datetime.now(UTC)
        settings = OrgBudgetSettings(
            org_id=budget_org.id,
            enabled=True,
            reset_day=1,
            monthly_limit=100.0,
            default_user_monthly_limit=30.0,
            slack_channel=None,
            slack_team_id=None,
            cycle_start_at=now,
            cycle_start_spend=20.0,
        )

        disabled_user_id = uuid4()
        override_user_id = uuid4()
        default_user_id = uuid4()

        overrides = [
            OrgUserBudgetOverride(
                org_id=budget_org.id,
                user_id=disabled_user_id,
                monthly_limit=None,
                is_disabled=True,
            ),
            OrgUserBudgetOverride(
                org_id=budget_org.id,
                user_id=override_user_id,
                monthly_limit=50.0,
                is_disabled=False,
            ),
        ]

        financial_data = {
            'members': {
                str(disabled_user_id): {'spend': 12.0},
                str(override_user_id): {'spend': 7.0},
                str(default_user_id): {'spend': 5.0},
            }
        }

        with (
            patch(
                'server.services.org_budget_service.LiteLlmManager.get_team_members_financial_data',
                AsyncMock(return_value=financial_data),
            ),
            patch(
                'server.services.org_budget_service.LiteLlmManager.update_team',
                AsyncMock(),
            ) as update_team,
            patch(
                'server.services.org_budget_service.LiteLlmManager.update_user_in_team',
                AsyncMock(),
            ) as update_user,
        ):
            await service._sync_litellm_budgets(budget_org.id, settings, overrides)

        update_team.assert_awaited_once_with(
            str(budget_org.id),
            team_alias=None,
            max_budget=120.0,
        )
        update_user.assert_has_awaits(
            [
                call(
                    str(disabled_user_id),
                    str(budget_org.id),
                    max_budget=None,
                    clear_budget=True,
                ),
                call(
                    str(override_user_id),
                    str(budget_org.id),
                    max_budget=57.0,
                    clear_budget=False,
                ),
                call(
                    str(default_user_id),
                    str(budget_org.id),
                    max_budget=35.0,
                    clear_budget=False,
                ),
            ],
            any_order=True,
        )

        assert settings.litellm_last_sync_status == 'success'
        assert settings.litellm_last_sync_error is None
        assert settings.litellm_last_sync_at is not None


@pytest.mark.asyncio
async def test_maybe_send_alerts_tracks_thresholds(async_session_maker, budget_org):
    async with async_session_maker() as session:
        service = OrgBudgetService(session)
        now = datetime.now(UTC)
        settings = OrgBudgetSettings(
            org_id=budget_org.id,
            enabled=True,
            reset_day=1,
            monthly_limit=100.0,
            default_user_monthly_limit=None,
            slack_channel=None,
            slack_team_id=None,
            cycle_start_at=now,
            cycle_start_spend=0.0,
        )
        cycle_start = now
        threshold_80 = OrgBudgetThreshold(
            org_id=budget_org.id,
            percentage=80,
            email_enabled=True,
            slack_enabled=False,
        )
        threshold_90 = OrgBudgetThreshold(
            org_id=budget_org.id,
            percentage=90,
            email_enabled=True,
            slack_enabled=False,
        )

        service.store.flush = AsyncMock()
        service._send_alerts = AsyncMock()

        await service._maybe_send_alerts(
            budget_org.id,
            settings,
            [threshold_80, threshold_90],
            current_spend=85.0,
            cycle_start=cycle_start,
        )

        service._send_alerts.assert_awaited_once()
        assert threshold_80.last_triggered_cycle_start == cycle_start
        assert threshold_80.last_triggered_at is not None
        assert threshold_90.last_triggered_at is None
        service.store.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_alerts_emails_and_slack(async_session_maker, budget_org):
    async with async_session_maker() as session:
        service = OrgBudgetService(session)
        settings = OrgBudgetSettings(
            org_id=budget_org.id,
            enabled=True,
            reset_day=1,
            monthly_limit=120.0,
            default_user_monthly_limit=None,
            slack_channel='alerts',
            slack_team_id='team-123',
            cycle_start_at=datetime.now(UTC),
            cycle_start_spend=0.0,
        )
        threshold = OrgBudgetThreshold(
            org_id=budget_org.id,
            percentage=90,
            email_enabled=True,
            slack_enabled=True,
        )

        service._get_org_name = AsyncMock(return_value='Acme')
        service._get_admin_emails = AsyncMock(return_value=['admin@example.com'])
        service._send_slack_alert = AsyncMock()

        with patch(
            'server.services.org_budget_service.SMTPEmailService.send_budget_alert_email',
            MagicMock(),
        ) as send_email:
            await service._send_alerts(
                budget_org.id,
                settings,
                threshold,
                current_spend=100.0,
                percentage=83.3,
            )

        send_email.assert_called_once_with(
            ['admin@example.com'],
            org_name='Acme',
            percentage=83.3,
            current_spend=100.0,
            monthly_limit=120.0,
            threshold=90,
        )
        service._send_slack_alert.assert_awaited_once_with(
            'Acme',
            settings,
            90,
            100.0,
            83.3,
        )
