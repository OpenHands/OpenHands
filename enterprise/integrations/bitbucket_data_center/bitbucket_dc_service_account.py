"""Posting-identity resolution for Bitbucket Data Center integrations.

Single source of truth for the credential that posts comments/reactions back to
Bitbucket DC: always the configured bot account, never the invoking user. A
user-authored reply containing "@openhands" would re-fire the webhook, so we
post only as the bot (mirroring the Jira DC service-account pattern). Callers
gate the event on ``BITBUCKET_DATA_CENTER_BOT_TOKEN`` before posting.
"""

from integrations.bitbucket_data_center.bitbucket_dc_service import (
    SaaSBitbucketDCService,
)
from pydantic import SecretStr
from server.auth.constants import BITBUCKET_DATA_CENTER_BOT_TOKEN


def bitbucket_dc_posting_service() -> SaaSBitbucketDCService:
    """Bitbucket DC service that posts as the configured bot account.

    Raises if the bot token is unset -- callers gate on it first, so this is a
    defensive check (we never fall back to the user's token).
    """
    if not BITBUCKET_DATA_CENTER_BOT_TOKEN:
        raise RuntimeError(
            'BITBUCKET_DATA_CENTER_BOT_TOKEN is not configured (required to '
            'post as the Bitbucket DC bot)'
        )
    # BBDC HTTP access tokens authenticate via Bearer. Set the raw token
    # directly so _get_headers uses Bearer -- the ``token=`` constructor arg
    # rewrites a colon-less token to ``x-token-auth:<token>`` (a Bitbucket
    # *Cloud* convention) sent as HTTP Basic, which Data Center 401s.
    service = SaaSBitbucketDCService()
    service.token = SecretStr(BITBUCKET_DATA_CENTER_BOT_TOKEN)
    return service
