"""Tests for the Bitbucket Data Center webhook route."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from server.routes.integration.bitbucket_dc import bitbucket_dc_events


def _signed(body: bytes, secret: str = 'shared-secret') -> str:
    return 'sha256=' + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _request_with_body(body: bytes) -> MagicMock:
    request = MagicMock()
    request.body = AsyncMock(return_value=body)
    return request


def _pr_comment_body() -> bytes:
    return json.dumps(
        {
            'pullRequest': {
                'id': 1,
                'toRef': {
                    'repository': {
                        'slug': 'myrepo',
                        'project': {'key': 'PROJ'},
                    }
                },
            },
            'comment': {'id': 99, 'text': 'Hey @openhands'},
        }
    ).encode()


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
@patch('server.routes.integration.bitbucket_dc.get_redis_client_async')
async def test_signature_verification_rejects_bad_signature_with_403(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_secret = AsyncMock(return_value='shared-secret')
    mock_manager.receive_message = AsyncMock()
    mock_get_redis_client_async.return_value = AsyncMock()

    body = _pr_comment_body()

    with pytest.raises(HTTPException) as exc:
        await bitbucket_dc_events(
            request=_request_with_body(body),
            x_hub_signature='sha256=deadbeef',
            x_event_key='pr:comment:added',
            x_request_id='req-1',
        )

    assert exc.value.status_code == 403
    mock_manager.receive_message.assert_not_called()


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
@patch('server.routes.integration.bitbucket_dc.get_redis_client_async')
async def test_missing_repo_identity_rejected_with_403(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_secret = AsyncMock(return_value='shared-secret')
    mock_manager.receive_message = AsyncMock()
    mock_get_redis_client_async.return_value = AsyncMock()

    body = json.dumps({'pullRequest': {'id': 1}}).encode()

    with pytest.raises(HTTPException) as exc:
        await bitbucket_dc_events(
            request=_request_with_body(body),
            x_hub_signature=_signed(body),
            x_event_key='pr:comment:added',
            x_request_id='req-1',
        )

    assert exc.value.status_code == 403
    mock_manager.receive_message.assert_not_called()


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
@patch('server.routes.integration.bitbucket_dc.get_redis_client_async')
async def test_duplicate_event_returns_200_and_skips_dispatch(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_secret = AsyncMock(return_value='shared-secret')
    mock_manager.receive_message = AsyncMock()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=False)  # duplicate
    mock_get_redis_client_async.return_value = redis

    body = _pr_comment_body()

    response = await bitbucket_dc_events(
        request=_request_with_body(body),
        x_hub_signature=_signed(body),
        x_event_key='pr:comment:added',
        x_request_id='req-1',
    )

    mock_manager.receive_message.assert_not_called()
    assert response.status_code == 200
    assert json.loads(response.body)['message'].startswith('Duplicate')


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
@patch('server.routes.integration.bitbucket_dc.get_redis_client_async')
async def test_valid_pr_comment_event_dispatches_to_manager_and_returns_200(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_secret = AsyncMock(return_value='shared-secret')
    mock_manager.receive_message = AsyncMock()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    mock_get_redis_client_async.return_value = redis

    body = _pr_comment_body()

    response = await bitbucket_dc_events(
        request=_request_with_body(body),
        x_hub_signature=_signed(body),
        x_event_key='pr:comment:added',
        x_request_id='req-1',
    )

    mock_manager.receive_message.assert_awaited_once()
    dispatched = mock_manager.receive_message.call_args.args[0]
    assert dispatched.source.value == 'bitbucket_data_center'
    assert dispatched.message['event_key'] == 'pr:comment:added'
    assert dispatched.message['installation_id'] == 'PROJ/myrepo'
    assert response.status_code == 200


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
async def test_diagnostics_ping_returns_200_without_dispatch(mock_manager):
    mock_manager.receive_message = AsyncMock()

    response = await bitbucket_dc_events(
        request=_request_with_body(b'{}'),
        x_hub_signature=None,
        x_event_key='diagnostics:ping',
        x_request_id='ping-1',
    )

    mock_manager.receive_message.assert_not_called()
    assert response.status_code == 200
