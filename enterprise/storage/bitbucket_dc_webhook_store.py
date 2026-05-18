from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import and_, or_, select, update
from storage.bitbucket_dc_webhook import BitbucketDCWebhook
from storage.database import a_session_maker


@dataclass
class BitbucketDCWebhookStore:
    """Read helpers for the ``bitbucket_dc_webhook`` table.

    Used by the Bitbucket Data Center webhook route to look up the
    per-repository secret needed to verify ``X-Hub-Signature``, and by
    the manager to resolve the keycloak ``user_id`` of the webhook
    installer.
    """

    async def get_webhook_secret(self, project_key: str, repo_slug: str) -> str | None:
        async with a_session_maker() as session:
            query = (
                select(BitbucketDCWebhook)
                .where(
                    BitbucketDCWebhook.project_key == project_key,
                    BitbucketDCWebhook.repo_slug == repo_slug,
                )
                .limit(1)
            )
            result = await session.execute(query)
            webhook = result.scalars().first()
            return webhook.webhook_secret if webhook else None

    async def get_webhook_user_id(self, project_key: str, repo_slug: str) -> str | None:
        async with a_session_maker() as session:
            query = (
                select(BitbucketDCWebhook.user_id)
                .where(
                    BitbucketDCWebhook.project_key == project_key,
                    BitbucketDCWebhook.repo_slug == repo_slug,
                )
                .limit(1)
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def get_webhook_by_repo(
        self, project_key: str, repo_slug: str
    ) -> BitbucketDCWebhook | None:
        async with a_session_maker() as session:
            query = (
                select(BitbucketDCWebhook)
                .where(
                    BitbucketDCWebhook.project_key == project_key,
                    BitbucketDCWebhook.repo_slug == repo_slug,
                )
                .limit(1)
            )
            result = await session.execute(query)
            return result.scalars().first()

    async def get_webhooks_by_repos(
        self, repos: list[tuple[str, str]]
    ) -> dict[tuple[str, str], BitbucketDCWebhook]:
        if not repos:
            return {}

        async with a_session_maker() as session:
            clauses = [
                and_(
                    BitbucketDCWebhook.project_key == project_key,
                    BitbucketDCWebhook.repo_slug == repo_slug,
                )
                for project_key, repo_slug in repos
            ]
            query = select(BitbucketDCWebhook).where(or_(*clauses))
            result = await session.execute(query)
            webhooks = result.scalars().all()
            return {
                (webhook.project_key, webhook.repo_slug): webhook
                for webhook in webhooks
            }

    async def upsert_webhook_enrollment(
        self,
        *,
        project_key: str,
        repo_slug: str,
        user_id: str,
        webhook_secret: str,
        webhook_id: str | None = None,
    ) -> BitbucketDCWebhook:
        async with a_session_maker() as session:
            async with session.begin():
                query = (
                    select(BitbucketDCWebhook)
                    .where(
                        BitbucketDCWebhook.project_key == project_key,
                        BitbucketDCWebhook.repo_slug == repo_slug,
                    )
                    .limit(1)
                )
                result = await session.execute(query)
                webhook = result.scalars().first()

                if webhook:
                    webhook.user_id = user_id
                    webhook.webhook_secret = webhook_secret
                    webhook.last_synced = datetime.now(timezone.utc)
                    if webhook_id is not None:
                        webhook.webhook_id = webhook_id
                else:
                    webhook = BitbucketDCWebhook(
                        project_key=project_key,
                        repo_slug=repo_slug,
                        user_id=user_id,
                        webhook_id=webhook_id,
                        webhook_secret=webhook_secret,
                    )
                    session.add(webhook)

            await session.refresh(webhook)
            return webhook

    async def update_webhook_id(
        self, *, project_key: str, repo_slug: str, webhook_id: str
    ) -> bool:
        async with a_session_maker() as session:
            async with session.begin():
                stmt = (
                    update(BitbucketDCWebhook)
                    .where(
                        BitbucketDCWebhook.project_key == project_key,
                        BitbucketDCWebhook.repo_slug == repo_slug,
                    )
                    .values(
                        webhook_id=webhook_id,
                        last_synced=datetime.now(timezone.utc),
                    )
                )
                result = await session.execute(stmt)
                return result.rowcount > 0

    @classmethod
    async def get_instance(cls) -> BitbucketDCWebhookStore:
        return BitbucketDCWebhookStore()
