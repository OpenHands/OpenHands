from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.engagement import Engagement, ScopeRule
from app.schemas.engagement import EngagementCreate, EngagementUpdate
from app.schemas.scope import AuthorizeScopeRequest, ScopeRuleCreate
from app.services.runtime_provisioner import RuntimeProvisioner
from app.services.scope_validator import is_target_allowed


class EngagementService:
    def __init__(
        self,
        db: AsyncSession,
        *,
        provisioner: RuntimeProvisioner | None = None,
    ):
        self.db = db
        self.provisioner = provisioner or RuntimeProvisioner()

    async def create(self, payload: EngagementCreate, *, created_by: str) -> Engagement:
        eng = Engagement(
            name=payload.name,
            client_name=payload.client_name,
            description=payload.description,
            runtime_profile=payload.runtime_profile,
            autonomy_mode=payload.autonomy_mode,
            status="draft",
            sandbox_status="stopped",
            created_by=created_by,
        )
        self.db.add(eng)
        await self.db.commit()
        await self.db.refresh(eng)
        return eng

    async def list_for_user(self, user_id: str) -> tuple[list[Engagement], int]:
        stmt = (
            select(Engagement)
            .where(Engagement.created_by == user_id)
            .order_by(Engagement.created_at.desc())
        )
        rows = list((await self.db.scalars(stmt)).all())
        return rows, len(rows)

    async def get(self, engagement_id: uuid.UUID, *, user_id: str | None = None) -> Engagement:
        eng = await self.db.get(Engagement, engagement_id)
        if eng is None:
            raise HTTPException(status_code=404, detail="Engagement not found")
        if user_id is not None and eng.created_by != user_id:
            raise HTTPException(status_code=404, detail="Engagement not found")
        return eng

    async def update(
        self, engagement_id: uuid.UUID, payload: EngagementUpdate, *, user_id: str
    ) -> tuple[Engagement, str | None]:
        """Update engagement fields.

        Returns ``(engagement, propagation)`` where ``propagation`` is set when
        ``autonomy_mode`` changes (PROJETOSIN-195):
        ``applied`` | ``pending_restart`` | ``n/a``.
        """
        eng = await self.get(engagement_id, user_id=user_id)
        changes = payload.model_dump(exclude_unset=True)
        autonomy_changed = "autonomy_mode" in changes
        for key, value in changes.items():
            setattr(eng, key, value)
        eng.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(eng)

        propagation: str | None = None
        if autonomy_changed:
            propagation = await self.propagate_autonomy_env(eng, user_id=user_id)
        return eng, propagation

    async def propagate_autonomy_env(
        self, eng: Engagement, *, user_id: str
    ) -> str:
        """Best-effort rewrite of compose env for ``PENTEST_AUTONOMY_MODE``.

        Live container recreate is deferred in MVP — when a compose project
        exists and the sandbox is running, return ``pending_restart`` so the UI
        can show the banner (AC-195-5).
        """
        if not eng.sandbox_compose_project:
            return "n/a"
        try:
            rules = await self.list_scope(eng.id, user_id=user_id)
            rewritten = await self.provisioner.rewrite_compose(eng, rules)
            if rewritten is None:
                return "n/a"
            if eng.sandbox_status == "running":
                return "pending_restart"
            return "n/a"
        except Exception:
            return "pending_restart"

    async def delete(self, engagement_id: uuid.UUID) -> None:
        eng = await self.get(engagement_id)
        await self.db.delete(eng)
        await self.db.commit()

    async def list_scope(self, engagement_id: uuid.UUID, *, user_id: str) -> list[ScopeRule]:
        await self.get(engagement_id, user_id=user_id)
        rows = (
            await self.db.scalars(
                select(ScopeRule).where(ScopeRule.engagement_id == engagement_id)
            )
        ).all()
        return list(rows)

    async def add_scope_rule(
        self, engagement_id: uuid.UUID, payload: ScopeRuleCreate, *, user_id: str
    ) -> ScopeRule:
        await self.get(engagement_id, user_id=user_id)
        rule = ScopeRule(
            engagement_id=engagement_id,
            rule_type=payload.rule_type,
            target_type=payload.target_type,
            target_value=payload.target_value,
            note=payload.note,
        )
        self.db.add(rule)
        await self.db.commit()
        await self.db.refresh(rule)
        return rule

    async def delete_scope_rule(
        self, engagement_id: uuid.UUID, rule_id: uuid.UUID, *, user_id: str
    ) -> None:
        await self.get(engagement_id, user_id=user_id)
        rule = await self.db.get(ScopeRule, rule_id)
        if rule is None or rule.engagement_id != engagement_id:
            raise HTTPException(status_code=404, detail="Scope rule not found")
        await self.db.delete(rule)
        await self.db.commit()

    async def authorize_scope(
        self, engagement_id: uuid.UUID, payload: AuthorizeScopeRequest, *, user_id: str
    ) -> Engagement:
        eng = await self.get(engagement_id, user_id=user_id)
        eng.scope_document_url = payload.scope_document_url
        eng.scope_authorized_at = datetime.now(timezone.utc)
        for rule in payload.scope_rules:
            self.db.add(
                ScopeRule(
                    engagement_id=engagement_id,
                    rule_type=rule.rule_type,
                    target_type=rule.target_type,
                    target_value=rule.target_value,
                    note=rule.note,
                )
            )
        eng.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(eng)
        return eng

    async def assert_workspace_ready(
        self, engagement_id: uuid.UUID, *, user_id: str
    ) -> Engagement:
        eng = await self.get(engagement_id, user_id=user_id)
        if eng.scope_authorized_at is None:
            raise HTTPException(
                status_code=400,
                detail="Scope not authorized; call authorize-scope first",
            )
        return eng

    async def provision(self, engagement_id: uuid.UUID, *, user_id: str) -> tuple[Engagement, uuid.UUID]:
        eng = await self.assert_workspace_ready(engagement_id, user_id=user_id)
        rules = await self.list_scope(engagement_id, user_id=user_id)
        eng.sandbox_status = "provisioning"
        project = await self.provisioner.provision(eng, rules)
        eng.sandbox_compose_project = project
        eng.sandbox_status = "running"
        eng.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(eng)
        return eng, uuid.uuid4()

    async def teardown(self, engagement_id: uuid.UUID, *, user_id: str) -> Engagement:
        eng = await self.get(engagement_id, user_id=user_id)
        await self.provisioner.teardown(eng)
        eng.sandbox_status = "stopped"
        eng.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(eng)
        return eng

    async def is_destination_allowed(
        self,
        engagement_id: uuid.UUID,
        *,
        target_type: str,
        target_value: str,
        user_id: str,
    ) -> bool:
        rules = await self.list_scope(engagement_id, user_id=user_id)
        tuples = [(r.rule_type, r.target_type, r.target_value) for r in rules]
        return is_target_allowed(
            tuples, target_type=target_type, target_value=target_value
        )
