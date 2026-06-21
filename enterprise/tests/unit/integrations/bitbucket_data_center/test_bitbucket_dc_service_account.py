"""Tests for the Bitbucket DC posting-identity service account."""

from unittest.mock import patch

import pytest
from integrations.bitbucket_data_center.bitbucket_dc_service_account import (
    bitbucket_dc_posting_service,
)


def test_posting_service_uses_bot_token_when_set():
    """Return a service authed as the bot when the token is configured."""
    with (
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.BITBUCKET_DATA_CENTER_BOT_TOKEN',
            'bot-pat-123',
        ),
        patch(
            'integrations.bitbucket_data_center.bitbucket_dc_service_account.SaaSBitbucketDCService'
        ) as service_cls,
    ):
        svc = bitbucket_dc_posting_service()

    # No per-user auth; the raw bot token is set directly so the service sends
    # Bearer (the token= ctor arg would downgrade to HTTP Basic, which DC 401s).
    service_cls.assert_called_once_with()
    assert 'external_auth_id' not in service_cls.call_args.kwargs
    assert svc.token.get_secret_value() == 'bot-pat-123'


def test_posting_service_raises_when_bot_token_unset():
    """Raise rather than fall back to a user token when no bot token is set."""
    with patch(
        'integrations.bitbucket_data_center.bitbucket_dc_service_account.BITBUCKET_DATA_CENTER_BOT_TOKEN',
        '',
    ):
        with pytest.raises(RuntimeError):
            bitbucket_dc_posting_service()
