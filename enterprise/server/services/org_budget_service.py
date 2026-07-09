from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import AsyncGenerator
from uuid import UUID

from fastapi import HTTPException, Request, status
from server.auth.authorization import RoleName
from server.services.smtp_email_service import SMTPEmailService
from sqlalchemy import and_, case, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.lite_llm_manager import LiteLlmManager
from storage.org import Org
from storage.org_budget_settings import OrgBudgetSettings
from storage.org_budget_store import OrgBudgetStore
from storage.org_budget_threshold import OrgBudgetThreshold
from storage.org_member import OrgMember
from storage.org_user_budget_override import OrgUserBudgetOverride
from storage.role import Role
from storage.slack_team import SlackTeam
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
from storage.user import User

from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.utils.logger import openhands_logger as logger

try:
    from slack_sdk.web.async_client import AsyncWebClient

    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False


DEFAULT_THRESHOLDS = (
    (80, True, False),
    (90, True, True),
    (100, True, True),
)


@dataclass
class BudgetCycle:
    start_at: datetime
    end_at: datetime


def _add_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _subtract_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def _current_cycle_start(now: datetime, reset_day: int) -> datetime:
    if now.day >= reset_day:
        return datetime(now.year, now.month, reset_day, tzinfo=UTC)
    prev_year, prev_month = _subtract_month(now.year, now.month)
    return datetime(prev_year, prev_month, reset_day, tzinfo=UTC)


def _next_cycle_start(cycle_start: datetime, reset_day: int) -> datetime:
    year, month = _add_month(cycle_start.year, cycle_start.month)
    return datetime(year, month, reset_day, tzinfo=UTC)


def _effective_user_budget_limit(
    override: OrgUserBudgetOverride | None,
    default_limit: float | None,
) -> tuple[float | None, bool, bool]:
    if override:
        if override.is_disabled:
            return None, True, True
        return override.monthly_limit, False, True
    return default_limit, False, False


def _escape_ilike(value: str) -> str:
    return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


class OrgBudgetService:
    def __init__(
        self,
        db_session: AsyncSession | None = None,
        store: OrgBudgetStore | None = None,
    ):
        if store is None:
            if db_session is None:
                raise ValueError('db_session is required when store is not provided')
            store = OrgBudgetStore(db_session=db_session)
        self.store = store
        self.db_session = store.db_session

    async def get_budget_state(
        self,
        org_id: UUID,
        users_page: int = 1,
        users_per_page: int = 50,
        users_search: str | None = None,
        users_status: str | None = None,
    ):
        settings = await self._get_or_create_settings(org_id)
        thresholds = await self._get_thresholds(org_id)
        cycle = self._current_cycle(settings)

        current_spend = await self._get_cycle_spend(org_id, cycle.start_at)
        users, users_total = await self._build_user_budget_rows(
            org_id,
            settings,
            cycle.start_at,
            users_page=users_page,
            users_per_page=users_per_page,
            users_search=users_search,
            users_status=users_status,
        )
        return {
            'settings': settings,
            'thresholds': thresholds,
            'cycle': cycle,
            'current_spend': current_spend,
            'users': users,
            'users_total': users_total,
            'users_page': users_page,
            'users_per_page': users_per_page,
        }

    async def run_budget_maintenance(self, org_id: UUID) -> dict:
        settings = await self._get_or_create_settings(org_id)
        thresholds = await self._get_thresholds(org_id)
        overrides = await self._get_overrides(org_id)
        cycle = self._current_cycle(settings)

        cycle_rolled = await self._roll_cycle_if_needed(settings, thresholds, overrides)
        if cycle_rolled:
            cycle = self._current_cycle(settings)

        current_spend = await self._get_cycle_spend(org_id, cycle.start_at)
        await self._maybe_send_alerts(
            org_id,
            settings,
            thresholds,
            current_spend,
            cycle.start_at,
        )
        if not cycle_rolled:
            await self._sync_litellm_budgets(org_id, settings, overrides)

        return {
            'cycle_start_at': cycle.start_at,
            'cycle_end_at': cycle.end_at,
            'cycle_rolled': cycle_rolled,
            'current_spend': current_spend,
        }

    async def update_budget_settings(
        self,
        org_id: UUID,
        update_data,
        users_page: int = 1,
        users_per_page: int = 50,
        users_search: str | None = None,
        users_status: str | None = None,
    ):
        settings = await self._get_or_create_settings(org_id)
        thresholds = await self._get_thresholds(org_id)
        overrides = await self._get_overrides(org_id)

        fields_set = update_data.model_fields_set
        reset_day_changed = False
        previous_enabled = settings.enabled

        if 'enabled' in fields_set:
            settings.enabled = update_data.enabled
        if 'monthly_limit' in fields_set:
            settings.monthly_limit = update_data.monthly_limit
        if 'reset_day' in fields_set:
            settings.reset_day = update_data.reset_day
            reset_day_changed = True
        if 'default_user_monthly_limit' in fields_set:
            settings.default_user_monthly_limit = update_data.default_user_monthly_limit
        if 'slack_channel' in fields_set:
            settings.slack_channel = update_data.slack_channel
        if 'slack_team_id' in fields_set:
            settings.slack_team_id = update_data.slack_team_id

        if settings.enabled and (
            settings.monthly_limit is None or settings.monthly_limit <= 0
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='monthly_limit is required when budgets are enabled',
            )

        if reset_day_changed or (not previous_enabled and settings.enabled):
            settings.cycle_start_at = _current_cycle_start(
                datetime.now(UTC), settings.reset_day
            )
            settings.cycle_start_spend = await self._fetch_team_spend(org_id)

        if 'thresholds' in fields_set and update_data.thresholds is not None:
            await self._replace_thresholds(org_id, thresholds, update_data.thresholds)
            thresholds = await self._get_thresholds(org_id)

        await self.store.flush()
        await self.store.refresh(settings)

        await self._sync_litellm_budgets(org_id, settings, overrides)

        cycle = self._current_cycle(settings)
        current_spend = await self._get_cycle_spend(org_id, cycle.start_at)
        users, users_total = await self._build_user_budget_rows(
            org_id,
            settings,
            cycle.start_at,
            users_page=users_page,
            users_per_page=users_per_page,
            users_search=users_search,
            users_status=users_status,
        )
        return {
            'settings': settings,
            'thresholds': thresholds,
            'cycle': cycle,
            'current_spend': current_spend,
            'users': users,
            'users_total': users_total,
            'users_page': users_page,
            'users_per_page': users_per_page,
        }

    async def upsert_user_override(
        self,
        org_id: UUID,
        user_id: UUID,
        monthly_limit: float | None,
        is_disabled: bool,
    ) -> OrgUserBudgetOverride:
        override = await self.store.upsert_override(
            org_id=org_id,
            user_id=user_id,
            monthly_limit=monthly_limit,
            is_disabled=is_disabled,
        )
        settings = await self._get_or_create_settings(org_id)
        overrides = await self._get_overrides(org_id)
        await self._sync_litellm_budgets(org_id, settings, overrides)
        return override

    async def delete_user_override(self, org_id: UUID, user_id: UUID) -> None:
        override = await self._get_override(org_id, user_id)
        if override is None:
            return
        await self.store.delete_override(override)
        settings = await self._get_or_create_settings(org_id)
        overrides = await self._get_overrides(org_id)
        await self._sync_litellm_budgets(org_id, settings, overrides)

    async def _get_or_create_settings(self, org_id: UUID) -> OrgBudgetSettings:
        settings = await self.store.get_settings(org_id)
        if settings:
            return settings

        return await self.store.create_settings(
            org_id=org_id,
            reset_day=1,
            cycle_start_at=_current_cycle_start(datetime.now(UTC), 1),
            thresholds=DEFAULT_THRESHOLDS,
        )

    async def _get_thresholds(self, org_id: UUID) -> list[OrgBudgetThreshold]:
        return await self.store.get_thresholds(org_id)

    async def _replace_thresholds(
        self,
        org_id: UUID,
        existing: list[OrgBudgetThreshold],
        new_thresholds,
    ) -> None:
        await self.store.replace_thresholds(org_id, existing, new_thresholds)

    async def _get_overrides(self, org_id: UUID) -> list[OrgUserBudgetOverride]:
        return await self.store.get_overrides(org_id)

    async def _get_override(
        self, org_id: UUID, user_id: UUID
    ) -> OrgUserBudgetOverride | None:
        return await self.store.get_override(org_id, user_id)

    def _current_cycle(self, settings: OrgBudgetSettings) -> BudgetCycle:
        start_at = settings.cycle_start_at
        end_at = _next_cycle_start(start_at, settings.reset_day)
        return BudgetCycle(start_at=start_at, end_at=end_at)

    async def _roll_cycle_if_needed(
        self,
        settings: OrgBudgetSettings,
        thresholds: list[OrgBudgetThreshold],
        overrides: list[OrgUserBudgetOverride],
    ) -> bool:
        now = datetime.now(UTC)
        next_cycle = _next_cycle_start(settings.cycle_start_at, settings.reset_day)
        if now < next_cycle:
            return False

        settings.cycle_start_at = _current_cycle_start(now, settings.reset_day)
        org_id = settings.org_id
        settings.cycle_start_spend = await self._fetch_team_spend(org_id)
        for threshold in thresholds:
            threshold.last_triggered_at = None
            threshold.last_triggered_cycle_start = None
        await self.store.flush()
        await self.store.refresh(settings)
        await self._sync_litellm_budgets(org_id, settings, overrides)
        return True

    async def _fetch_team_spend(self, org_id: UUID) -> float:
        try:
            financial_data = await LiteLlmManager.get_team_members_financial_data(
                str(org_id)
            )
        except Exception as e:
            logger.warning(
                'org_budget_team_spend_fetch_failed',
                extra={'org_id': str(org_id), 'error': str(e)},
            )
            return 0.0
        if not financial_data:
            return 0.0
        return float(financial_data.get('team_spend') or 0.0)

    async def _get_cycle_spend(self, org_id: UUID, cycle_start: datetime) -> float:
        total_query = (
            select(
                func.coalesce(func.sum(StoredConversationMetadata.accumulated_cost), 0)
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(StoredConversationMetadata.conversation_version == 'V1')
            .where(StoredConversationMetadataSaas.org_id == org_id)
            .where(StoredConversationMetadata.created_at >= cycle_start)
        )
        result = await self.db_session.execute(total_query)
        return float(result.scalar() or 0.0)

    async def _build_user_budget_rows(
        self,
        org_id: UUID,
        settings: OrgBudgetSettings,
        cycle_start: datetime,
        users_page: int,
        users_per_page: int,
        users_search: str | None,
        users_status: str | None,
    ) -> tuple[list[dict], int]:
        spend_subquery = (
            select(
                StoredConversationMetadataSaas.user_id.label('user_id'),
                func.coalesce(
                    func.sum(StoredConversationMetadata.accumulated_cost), 0
                ).label('current_spend'),
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(StoredConversationMetadata.conversation_version == 'V1')
            .where(StoredConversationMetadataSaas.org_id == org_id)
            .where(StoredConversationMetadata.created_at >= cycle_start)
            .group_by(StoredConversationMetadataSaas.user_id)
        ).subquery()

        default_limit = literal(settings.default_user_monthly_limit)
        effective_limit = case(
            (OrgUserBudgetOverride.is_disabled.is_(True), literal(None)),
            (
                OrgUserBudgetOverride.user_id.isnot(None),
                OrgUserBudgetOverride.monthly_limit,
            ),
            else_=default_limit,
        )
        is_disabled = case(
            (OrgUserBudgetOverride.is_disabled.is_(True), literal(True)),
            else_=literal(False),
        )
        is_override = OrgUserBudgetOverride.user_id.isnot(None)
        current_spend = func.coalesce(spend_subquery.c.current_spend, 0.0)

        base_query = (
            select(
                OrgMember.user_id.label('user_id'),
                User.email.label('user_email'),
                User.git_user_name.label('user_name'),
                current_spend.label('current_spend'),
                OrgUserBudgetOverride.monthly_limit.label('monthly_limit'),
                effective_limit.label('effective_monthly_limit'),
                is_disabled.label('is_disabled'),
                is_override.label('is_override'),
            )
            .select_from(OrgMember)
            .join(User, OrgMember.user_id == User.id)
            .outerjoin(
                OrgUserBudgetOverride,
                and_(
                    OrgUserBudgetOverride.org_id == org_id,
                    OrgUserBudgetOverride.user_id == OrgMember.user_id,
                ),
            )
            .outerjoin(spend_subquery, spend_subquery.c.user_id == OrgMember.user_id)
            .where(OrgMember.org_id == org_id)
        )

        search_value = (users_search or '').strip()
        if search_value:
            escaped = _escape_ilike(search_value)
            pattern = f'%{escaped}%'
            base_query = base_query.where(
                or_(
                    User.email.ilike(pattern, escape='\\'),
                    User.git_user_name.ilike(pattern, escape='\\'),
                )
            )

        status_value = (users_status or '').strip().lower()
        if status_value:
            has_limit = and_(
                is_disabled.is_(False),
                effective_limit.isnot(None),
                effective_limit > 0,
            )
            if status_value == 'disabled':
                base_query = base_query.where(is_disabled.is_(True))
            elif status_value == 'nocap':
                base_query = base_query.where(
                    and_(
                        is_disabled.is_(False),
                        or_(effective_limit.is_(None), effective_limit <= 0),
                    )
                )
            elif status_value == 'overcap':
                base_query = base_query.where(
                    and_(has_limit, current_spend > effective_limit)
                )
            elif status_value == 'over90':
                base_query = base_query.where(
                    and_(has_limit, current_spend >= effective_limit * 0.9)
                )
            elif status_value == 'over80':
                base_query = base_query.where(
                    and_(has_limit, current_spend >= effective_limit * 0.8)
                )
            elif status_value == 'ontrack':
                base_query = base_query.where(
                    and_(has_limit, current_spend < effective_limit * 0.8)
                )

        total_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.db_session.execute(total_query)
        total = int(total_result.scalar() or 0)

        offset = (users_page - 1) * users_per_page
        query = (
            base_query.order_by(User.email.asc(), User.id.asc())
            .limit(users_per_page)
            .offset(offset)
        )
        result = await self.db_session.execute(query)
        rows = []
        for row in result:
            rows.append(
                {
                    'user_id': str(row.user_id),
                    'user_email': row.user_email,
                    'user_name': row.user_name,
                    'current_spend': float(row.current_spend or 0.0),
                    'monthly_limit': row.monthly_limit,
                    'effective_monthly_limit': row.effective_monthly_limit,
                    'is_disabled': bool(row.is_disabled),
                    'is_override': bool(row.is_override),
                }
            )
        return rows, total

    async def _get_user_spend(
        self, org_id: UUID, user_id: UUID, cycle_start: datetime
    ) -> float:
        user_query = (
            select(
                func.coalesce(func.sum(StoredConversationMetadata.accumulated_cost), 0)
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(StoredConversationMetadata.conversation_version == 'V1')
            .where(StoredConversationMetadataSaas.org_id == org_id)
            .where(StoredConversationMetadataSaas.user_id == user_id)
            .where(StoredConversationMetadata.created_at >= cycle_start)
        )
        result = await self.db_session.execute(user_query)
        return float(result.scalar() or 0.0)

    async def get_user_budget_row(self, org_id: UUID, user_id: UUID) -> dict | None:
        settings = await self._get_or_create_settings(org_id)
        overrides = await self._get_overrides(org_id)
        cycle = self._current_cycle(settings)

        result = await self.db_session.execute(
            select(OrgMember, User)
            .join(User, OrgMember.user_id == User.id)
            .where(OrgMember.org_id == org_id)
            .where(OrgMember.user_id == user_id)
        )
        row = result.one_or_none()
        if not row:
            return None

        org_member, user = row
        override = next(
            (override for override in overrides if override.user_id == user_id),
            None,
        )
        effective_limit, is_disabled, is_override = _effective_user_budget_limit(
            override, settings.default_user_monthly_limit
        )
        current_spend = await self._get_user_spend(org_id, user_id, cycle.start_at)
        return {
            'user_id': str(org_member.user_id),
            'user_email': user.email,
            'user_name': user.git_user_name,
            'current_spend': current_spend,
            'monthly_limit': override.monthly_limit if override else None,
            'effective_monthly_limit': effective_limit,
            'is_disabled': is_disabled,
            'is_override': is_override,
        }

    async def _record_litellm_sync(
        self,
        settings: OrgBudgetSettings,
        status_value: str,
        error: str | None = None,
    ) -> None:
        settings.litellm_last_sync_at = datetime.now(UTC)
        settings.litellm_last_sync_status = status_value
        settings.litellm_last_sync_error = error
        await self.store.flush()

    async def _sync_litellm_budgets(
        self,
        org_id: UUID,
        settings: OrgBudgetSettings,
        overrides: list[OrgUserBudgetOverride],
    ) -> None:
        sync_errors: list[str] = []
        try:
            financial_data = await LiteLlmManager.get_team_members_financial_data(
                str(org_id)
            )
        except Exception as e:
            error_message = f'fetch_failed: {e}'
            logger.warning(
                'org_budget_litellm_fetch_failed',
                extra={'org_id': str(org_id), 'error': str(e)},
            )
            await self._record_litellm_sync(settings, 'error', error_message[:500])
            return

        if not financial_data:
            await self._record_litellm_sync(settings, 'skipped')
            return

        members = financial_data.get('members', {})

        if settings.enabled and settings.monthly_limit:
            # Anchor LiteLLM budgets to our cycle start spend so resets stay aligned.
            max_budget = settings.cycle_start_spend + settings.monthly_limit
            try:
                await LiteLlmManager.update_team(
                    str(org_id),
                    team_alias=None,
                    max_budget=max_budget,
                )
            except Exception as e:
                sync_errors.append(f'team_update_failed: {e}')
                logger.warning(
                    'org_budget_litellm_team_update_failed',
                    extra={'org_id': str(org_id), 'error': str(e)},
                )
        else:
            try:
                await LiteLlmManager.update_team(
                    str(org_id),
                    team_alias=None,
                    max_budget=None,
                    clear_budget=True,
                )
            except Exception as e:
                sync_errors.append(f'team_clear_failed: {e}')
                logger.warning(
                    'org_budget_litellm_team_clear_failed',
                    extra={'org_id': str(org_id), 'error': str(e)},
                )

        override_map = {str(o.user_id): o for o in overrides}

        for user_id, info in members.items():
            override = override_map.get(user_id)
            effective_limit, is_disabled, _ = _effective_user_budget_limit(
                override, settings.default_user_monthly_limit
            )
            if is_disabled:
                max_budget_in_team = None
                clear_budget = True
            elif effective_limit is not None:
                max_budget_in_team = float(info.get('spend') or 0.0) + effective_limit
                clear_budget = False
            else:
                max_budget_in_team = None
                clear_budget = True
            try:
                await LiteLlmManager.update_user_in_team(
                    user_id,
                    str(org_id),
                    max_budget=max_budget_in_team,
                    clear_budget=clear_budget,
                )
            except Exception as e:
                sync_errors.append(f'user_update_failed: {user_id}: {e}')
                logger.warning(
                    'org_budget_litellm_user_update_failed',
                    extra={
                        'org_id': str(org_id),
                        'user_id': user_id,
                        'error': str(e),
                    },
                )

        if sync_errors:
            summary = sync_errors[0]
            if len(sync_errors) > 1:
                summary = f'{summary} (+{len(sync_errors) - 1} more)'
            await self._record_litellm_sync(settings, 'error', summary[:500])
        else:
            await self._record_litellm_sync(settings, 'success')

    async def _maybe_send_alerts(
        self,
        org_id: UUID,
        settings: OrgBudgetSettings,
        thresholds: list[OrgBudgetThreshold],
        current_spend: float,
        cycle_start: datetime,
    ) -> None:
        if not settings.enabled or not settings.monthly_limit:
            return

        if settings.monthly_limit <= 0:
            return

        percentage = (current_spend / settings.monthly_limit) * 100
        now = datetime.now(UTC)

        triggered = False
        for threshold in thresholds:
            if percentage < threshold.percentage:
                continue
            if threshold.last_triggered_cycle_start == cycle_start:
                continue

            await self._send_alerts(
                org_id,
                settings,
                threshold,
                current_spend,
                percentage,
            )
            threshold.last_triggered_at = now
            threshold.last_triggered_cycle_start = cycle_start
            triggered = True

        if triggered:
            await self.store.flush()

    async def _send_alerts(
        self,
        org_id: UUID,
        settings: OrgBudgetSettings,
        threshold: OrgBudgetThreshold,
        current_spend: float,
        percentage: float,
    ) -> None:
        org_name = await self._get_org_name(org_id)
        if threshold.email_enabled:
            recipients = await self._get_admin_emails(org_id)
            if recipients:
                await asyncio.to_thread(
                    SMTPEmailService.send_budget_alert_email,
                    recipients,
                    org_name=org_name,
                    percentage=percentage,
                    current_spend=current_spend,
                    monthly_limit=settings.monthly_limit or 0,
                    threshold=threshold.percentage,
                )

        if threshold.slack_enabled:
            await self._send_slack_alert(
                org_name,
                settings,
                threshold.percentage,
                current_spend,
                percentage,
            )

    async def _get_org_name(self, org_id: UUID) -> str:
        result = await self.db_session.execute(select(Org.name).where(Org.id == org_id))
        return result.scalar_one_or_none() or 'your organization'

    async def _get_admin_emails(self, org_id: UUID) -> list[str]:
        query = (
            select(User.email)
            .join(OrgMember, OrgMember.user_id == User.id)
            .join(Role, Role.id == OrgMember.role_id)
            .where(OrgMember.org_id == org_id)
            .where(Role.name.in_([RoleName.ADMIN.value, RoleName.OWNER.value]))
        )
        result = await self.db_session.execute(query)
        return [row.email for row in result if row.email]

    async def _send_slack_alert(
        self,
        org_name: str,
        settings: OrgBudgetSettings,
        threshold: int,
        current_spend: float,
        percentage: float,
    ) -> None:
        if not SLACK_AVAILABLE:
            logger.warning('Slack SDK not installed, skipping slack budget alert')
            return
        if not settings.slack_channel:
            return
        team_id = await self._resolve_slack_team_id(settings)
        if not team_id:
            return
        result = await self.db_session.execute(
            select(SlackTeam.bot_access_token).where(SlackTeam.team_id == team_id)
        )
        token = result.scalar_one_or_none()
        if not token:
            return

        client = AsyncWebClient(token=token)
        message = (
            f':warning: OpenHands budget alert for *{org_name}*\n'
            f'Threshold: *{threshold}%*\n'
            f'Current spend: *${current_spend:,.2f}* '
            f'({percentage:.1f}% of ${settings.monthly_limit:,.2f})'
        )
        try:
            await client.chat_postMessage(
                channel=settings.slack_channel,
                text=message,
            )
        except Exception as e:
            logger.warning(
                'Slack budget alert failed',
                extra={'error': str(e), 'team_id': team_id},
            )

    async def _resolve_slack_team_id(self, settings: OrgBudgetSettings) -> str | None:
        if settings.slack_team_id:
            return settings.slack_team_id
        result = await self.db_session.execute(select(SlackTeam.team_id))
        team_ids = [row.team_id for row in result]
        if len(team_ids) == 1:
            return team_ids[0]
        if team_ids:
            logger.warning(
                'Multiple Slack teams configured; set slack_team_id to enable alerts'
            )
        return None


class OrgBudgetServiceInjector(Injector[OrgBudgetService]):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[OrgBudgetService, None]:
        from openhands.app_server.config import get_db_session

        async with get_db_session(state, request) as db_session:
            yield OrgBudgetService(db_session=db_session)
