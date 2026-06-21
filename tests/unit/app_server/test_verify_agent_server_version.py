"""Tests for the agent-server SDK version check on conversation start.

Covers _verify_agent_server_version (only enforced for custom sandbox images,
fail-open on anything it can't read) and the is_custom_agent_server_image gate.
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from openhands.app_server.app_conversation.live_status_app_conversation_service import (
    LiveStatusAppConversationService,
    _expected_sdk_version,
)
from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.sandbox_spec_service import (
    AGENT_SERVER_IMAGE,
    is_custom_agent_server_image,
)

MODULE = 'openhands.app_server.app_conversation.live_status_app_conversation_service'
PINNED_TAG = AGENT_SERVER_IMAGE.rsplit(':', 1)[-1]
# A custom pin = a tag different from the release-default one.
CUSTOM_ENV = {
    'AGENT_SERVER_IMAGE_REPOSITORY': 'harbor.example/agent-server',
    'AGENT_SERVER_IMAGE_TAG': 'custom-9',
}


def _service(client):
    # Only httpx_client is touched by the method under test.
    svc = LiveStatusAppConversationService.__new__(LiveStatusAppConversationService)
    svc.httpx_client = client
    return svc


def _resp(sdk_version):
    return Mock(
        raise_for_status=Mock(),
        json=Mock(return_value={'sdk_version': sdk_version}),
    )


async def _verify(client, env):
    with patch.dict(os.environ, env, clear=False):
        await _service(client)._verify_agent_server_version('http://agent.test/', 'k')


@pytest.mark.asyncio
async def test_default_image_skips_check():
    client = Mock(get=AsyncMock())
    await _verify(client, {'AGENT_SERVER_IMAGE_TAG': PINNED_TAG})
    client.get.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('val', ['1', 'true', 'TRUE', 'yes'])
async def test_opt_out_skips_check(val):
    client = Mock(get=AsyncMock())
    await _verify(client, {**CUSTOM_ENV, 'OH_SKIP_AGENT_SERVER_VERSION_CHECK': val})
    client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_matching_version_passes():
    expected = _expected_sdk_version()
    assert expected
    client = Mock(get=AsyncMock(return_value=_resp(expected)))
    await _verify(client, CUSTOM_ENV)
    client.get.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('suffix', ['.99', '.0+c24'])
async def test_patch_or_local_retag_tolerated(suffix):
    major, minor = _expected_sdk_version().split('.')[:2]
    client = Mock(get=AsyncMock(return_value=_resp(f'{major}.{minor}{suffix}')))
    await _verify(client, CUSTOM_ENV)  # major.minor equal -> no raise


@pytest.mark.asyncio
async def test_minor_mismatch_raises():
    expected = _expected_sdk_version()
    reported = f'{int(expected.split(".")[0]) + 1}.0.0'
    client = Mock(get=AsyncMock(return_value=_resp(reported)))
    with pytest.raises(SandboxError) as ei:
        await _verify(client, CUSTOM_ENV)
    # str() of a SandboxError (HTTPException) is '500: <msg>'; assert substrings.
    assert reported in str(ei.value)
    assert 'OH_SKIP_AGENT_SERVER_VERSION_CHECK' in str(ei.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('reported', ['unknown', '', 'not-a-version'])
async def test_unverifiable_versions_pass(reported):
    client = Mock(get=AsyncMock(return_value=_resp(reported)))
    await _verify(client, CUSTOM_ENV)


@pytest.mark.asyncio
async def test_non_200_server_info_fails_open():
    resp = Mock(
        raise_for_status=Mock(
            side_effect=httpx.HTTPStatusError(
                'x', request=Mock(), response=Mock(status_code=404)
            )
        )
    )
    client = Mock(get=AsyncMock(return_value=resp))
    await _verify(client, CUSTOM_ENV)
    client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_transport_error_fails_open():
    client = Mock(get=AsyncMock(side_effect=httpx.ConnectError('down')))
    await _verify(client, CUSTOM_ENV)


@pytest.mark.asyncio
async def test_no_expected_version_skips_request():
    client = Mock(get=AsyncMock())
    with patch(f'{MODULE}._expected_sdk_version', return_value=None):
        await _verify(client, CUSTOM_ENV)
    client.get.assert_not_awaited()


def test_is_custom_agent_server_image():
    with patch.dict(os.environ, {}, clear=True):
        assert is_custom_agent_server_image() is False
    with patch.dict(os.environ, {'AGENT_SERVER_IMAGE_TAG': PINNED_TAG}, clear=False):
        assert is_custom_agent_server_image() is False
    with patch.dict(os.environ, {'AGENT_SERVER_IMAGE_TAG': 'custom-9'}, clear=False):
        assert is_custom_agent_server_image() is True
