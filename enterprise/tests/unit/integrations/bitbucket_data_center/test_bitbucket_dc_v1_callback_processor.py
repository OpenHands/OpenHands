"""Tests for the Bitbucket DC v1 callback processor summary posting."""

from unittest.mock import AsyncMock, patch

import pytest
from integrations.bitbucket_data_center.bitbucket_dc_v1_callback_processor import (
    BitbucketDCV1CallbackProcessor,
)


@pytest.mark.asyncio
async def test_post_summary_uses_bot_posting_service():
    """Summary comments post via the bot posting service, not a user token."""
    processor = BitbucketDCV1CallbackProcessor(
        bitbucket_dc_view_data={
            'project_key': 'PROJ',
            'repo_slug': 'myrepo',
            'pr_id': 7,
            'parent_comment_id': 42,
        }
    )
    fake_service = AsyncMock()

    with patch(
        'integrations.bitbucket_data_center.bitbucket_dc_service_account.bitbucket_dc_posting_service',
        return_value=fake_service,
    ) as mock_posting:
        await processor._post_summary_to_bitbucket_dc('All done!')

    mock_posting.assert_called_once_with()
    fake_service.reply_to_pr_comment.assert_awaited_once_with(
        owner='PROJ',
        repo_slug='myrepo',
        pr_id=7,
        body='All done!',
        parent_comment_id=42,
    )
