from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from storage.bitbucket_webhook import BitbucketWebhook
from storage.database import a_session_maker


@dataclass
class BitbucketWebhookStore:
    """Read/write helpers for the ``bitbucket_webhook`` table.

    Used by the resolver webhook route to look up the per-installation
    secret needed to verify ``X-Hub-Signature``.
    """

    async def get_webhook_secret(self, webhook_uuid: str) -> str | None:
        """Return the shared secret for ``webhook_uuid``, or None."""
        async with a_session_maker() as session:
            query = (
                select(BitbucketWebhook)
                .where(BitbucketWebhook.webhook_uuid == webhook_uuid)
                .limit(1)
            )
            result = await session.execute(query)
            webhook = result.scalars().first()
            return webhook.webhook_secret if webhook else None

    async def get_webhook_user_id(self, webhook_uuid: str) -> str | None:
        """Return the keycloak ``user_id`` of the installer for
        ``webhook_uuid``, or None.

        Used by :class:`BitbucketManager` to resolve "who installed this
        webhook?" without depending on a per-actor Keycloak attribute
        mapper (which Bitbucket Cloud's built-in IdP cannot populate).
        """
        async with a_session_maker() as session:
            query = (
                select(BitbucketWebhook.user_id)
                .where(BitbucketWebhook.webhook_uuid == webhook_uuid)
                .limit(1)
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()

    @classmethod
    async def get_instance(cls) -> BitbucketWebhookStore:
        return BitbucketWebhookStore()
