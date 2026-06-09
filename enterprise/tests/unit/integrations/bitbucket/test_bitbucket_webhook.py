"""Tests for the Bitbucket Cloud webhook route."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from server.routes.integration.bitbucket import bitbucket_events


def _signed(body: bytes, secret: str = 'shared-secret') -> str:
    return 'sha256=' + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _request_with_body(body: bytes) -> MagicMock:
    request = MagicMock()
    request.body = AsyncMock(return_value=body)
    return request


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket.webhook_store')
@patch('server.routes.integration.bitbucket.bitbucket_manager')
@patch('server.routes.integration.bitbucket.get_redis_client_async')
async def test_missing_hook_uuid_header_rejected_with_403(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_secret = AsyncMock(return_value='shared-secret')
    mock_manager.receive_message = AsyncMock()
    mock_get_redis_client_async.return_value = AsyncMock()
    body = json.dumps({'pullrequest': {'id': 1}}).encode()

    with pytest.raises(HTTPException) as exc:
        await bitbucket_events(
            request=_request_with_body(body),
            x_hub_signature=_signed(body),
            x_event_key='pullrequest:comment_created',
            x_request_uuid='req-1',
            x_hook_uuid=None,
        )

    assert exc.value.status_code == 403
    mock_manager.receive_message.assert_not_called()


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket.webhook_store')
@patch('server.routes.integration.bitbucket.bitbucket_manager')
@patch('server.routes.integration.bitbucket.get_redis_client_async')
async def test_signature_verification_rejects_bad_signature_with_403(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_secret = AsyncMock(return_value='shared-secret')
    mock_manager.receive_message = AsyncMock()
    mock_get_redis_client_async.return_value = AsyncMock()
    body = json.dumps({'pullrequest': {'id': 1}}).encode()

    with pytest.raises(HTTPException) as exc:
        await bitbucket_events(
            request=_request_with_body(body),
            x_hub_signature='sha256=deadbeef',
            x_event_key='pullrequest:comment_created',
            x_request_uuid='req-1',
            x_hook_uuid='install-uuid',
        )

    assert exc.value.status_code == 403
    mock_manager.receive_message.assert_not_called()


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket.webhook_store')
@patch('server.routes.integration.bitbucket.bitbucket_manager')
@patch('server.routes.integration.bitbucket.get_redis_client_async')
async def test_duplicate_event_returns_200_and_skips_dispatch(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_secret = AsyncMock(return_value='shared-secret')
    mock_manager.receive_message = AsyncMock()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=False)  # duplicate
    mock_get_redis_client_async.return_value = redis

    body = json.dumps({'pullrequest': {'id': 1}, 'comment': {'id': 99}}).encode()

    response = await bitbucket_events(
        request=_request_with_body(body),
        x_hub_signature=_signed(body),
        x_event_key='pullrequest:comment_created',
        x_request_uuid='req-1',
        x_hook_uuid='install-uuid',
    )

    mock_manager.receive_message.assert_not_called()
    assert response.status_code == 200
    assert json.loads(response.body)['message'].startswith('Duplicate')


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket.webhook_store')
@patch('server.routes.integration.bitbucket.bitbucket_manager')
@patch('server.routes.integration.bitbucket.get_redis_client_async')
async def test_valid_event_dispatches_to_manager_and_returns_200(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_secret = AsyncMock(return_value='shared-secret')
    mock_manager.receive_message = AsyncMock()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    mock_get_redis_client_async.return_value = redis

    body = json.dumps({'pullrequest': {'id': 1}, 'comment': {'id': 99}}).encode()

    response = await bitbucket_events(
        request=_request_with_body(body),
        x_hub_signature=_signed(body),
        x_event_key='pullrequest:comment_created',
        x_request_uuid='req-1',
        x_hook_uuid='install-uuid',
    )

    mock_manager.receive_message.assert_awaited_once()
    dispatched = mock_manager.receive_message.call_args.args[0]
    assert dispatched.source.value == 'bitbucket'
    assert dispatched.message['event_key'] == 'pullrequest:comment_created'
    assert response.status_code == 200
