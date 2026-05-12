from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
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

    @classmethod
    async def get_instance(cls) -> BitbucketDCWebhookStore:
        return BitbucketDCWebhookStore()
