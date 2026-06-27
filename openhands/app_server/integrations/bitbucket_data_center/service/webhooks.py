from typing import Any

from openhands.app_server.integrations.bitbucket_data_center.service.base import (
    BitbucketDCMixinBase,
)
from openhands.app_server.integrations.service_types import RequestMethod


class BitbucketDCWebhooksMixin(BitbucketDCMixinBase):
    """Repository webhook helpers for Bitbucket Data Center.

    Direct analog of ``BitBucketWebhooksMixin`` (Bitbucket Cloud), hitting
    BBDC's REST API for webhook CRUD instead of the manual paste-the-secret
    flow that the initial enrollment UI required. BBDC's webhook endpoints
    sit under ``/rest/api/1.0/projects/{key}/repos/{slug}/webhooks[/{id}]``
    and require the caller's token to have ``REPO_ADMIN`` scope — the
    Keycloak BBDC IDP requests this scope by default once webhook lifecycle
    is enabled.

    Notes on BBDC vs Cloud:
      * BBDC's webhook id is a numeric ``int`` (not a UUID). It is exposed
        here as ``str`` to match the ``bitbucket_dc_webhook.webhook_id``
        column type.
      * BBDC's payload puts the shared secret inside ``configuration``,
        not at the top level (Cloud uses ``secret``).
      * BBDC's pagination shape is ``{"values": [...], "isLastPage": ...}``
        — the inherited ``_fetch_paginated_data`` helper from
        :class:`BitbucketDCMixinBase` already handles that.
    """

    async def get_repository_webhooks(
        self, owner: str, repo_slug: str
    ) -> list[dict[str, Any]]:
        url = f'{self._repo_api_base(owner, repo_slug)}/webhooks'
        return await self._fetch_paginated_data(url, {'limit': 100}, max_items=100)

    async def check_webhook_exists_on_repository(
        self, owner: str, repo_slug: str, webhook_url: str
    ) -> tuple[bool, str | None]:
        """Return ``(True, id)`` when a webhook with ``webhook_url`` already
        exists on the repo (used to make enroll/reinstall idempotent), else
        ``(False, None)``.

        Webhook ``id`` is normalized to ``str`` to match the storage column
        type, which keeps callers from having to coerce when persisting.
        """
        webhooks = await self.get_repository_webhooks(owner, repo_slug)
        for webhook in webhooks:
            if webhook.get('url') == webhook_url:
                wid = webhook.get('id')
                return True, str(wid) if wid is not None else None
        return False, None

    async def create_repository_webhook(
        self,
        *,
        owner: str,
        repo_slug: str,
        name: str,
        webhook_url: str,
        webhook_secret: str,
        events: list[str],
    ) -> str | None:
        """Create a webhook via ``POST .../webhooks``. Returns the new id."""
        url = f'{self._repo_api_base(owner, repo_slug)}/webhooks'
        payload = {
            'name': name,
            'url': webhook_url,
            'active': True,
            'events': events,
            'configuration': {'secret': webhook_secret},
        }
        response, _ = await self._make_request(
            url=url,
            params=payload,
            method=RequestMethod.POST,
        )
        if not response:
            return None
        wid = response.get('id')
        return str(wid) if wid is not None else None

    async def update_repository_webhook(
        self,
        *,
        owner: str,
        repo_slug: str,
        webhook_id: str,
        name: str,
        webhook_url: str,
        webhook_secret: str,
        events: list[str],
    ) -> str | None:
        """Update an existing webhook via ``PUT .../webhooks/{id}``.

        BBDC requires the full payload on PUT (not partial), so we send the
        same body shape as ``create_repository_webhook``.
        """
        url = f'{self._repo_api_base(owner, repo_slug)}/webhooks/{webhook_id}'
        payload = {
            'name': name,
            'url': webhook_url,
            'active': True,
            'events': events,
            'configuration': {'secret': webhook_secret},
        }
        response, _ = await self._make_request(
            url=url,
            params=payload,
            method=RequestMethod.PUT,
        )
        if not response:
            return webhook_id
        wid = response.get('id')
        return str(wid) if wid is not None else webhook_id

    async def delete_repository_webhook(
        self, owner: str, repo_slug: str, webhook_id: str
    ) -> None:
        url = f'{self._repo_api_base(owner, repo_slug)}/webhooks/{webhook_id}'
        await self._make_request(url=url, method=RequestMethod.DELETE)

    async def user_has_admin_access(self, owner: str, repo_slug: str) -> bool:
        """Best-effort admin probe — trusts the OAuth scope grant.

        BBDC's permissions-introspection endpoints all require ``REPO_ADMIN``
        to query, and on the SaaS path ``self.user_id`` is unset (the OAuth
        bearer token doesn't carry a BBDC slug), so a strict pre-check has
        no reliable identifier to look up. The OAuth scope grant itself is
        the gating signal: if the user got a token with ``REPO_ADMIN`` they
        had to consent to it, and the eventual webhook write will surface
        any *real* authorization problem with an actionable BBDC error.

        Same philosophy as ``user_has_write_access_for`` (see its long
        docstring in ``resolver.py``) — trust the scope, let the API call
        be the authoritative check.
        """
        return True
