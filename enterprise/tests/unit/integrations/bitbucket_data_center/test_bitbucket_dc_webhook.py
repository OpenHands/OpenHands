"""Tests for the Bitbucket Data Center webhook route."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from server.routes.integration.bitbucket_dc import (
    bitbucket_dc_events,
    enroll_bitbucket_dc_webhook,
    get_bitbucket_dc_resources,
    update_bitbucket_dc_webhook_id,
)

from openhands.app_server.integrations.service_types import ProviderType, Repository


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


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.SaaSBitbucketDCService')
async def test_get_bitbucket_dc_resources_returns_repo_enrollment_status(
    mock_service_cls, mock_store
):
    service = MagicMock()
    service.get_all_repositories = AsyncMock(
        return_value=[
            Repository(
                id='1',
                full_name='PROJ/myrepo',
                git_provider=ProviderType.BITBUCKET_DATA_CENTER,
                is_public=False,
            )
        ]
    )
    mock_service_cls.return_value = service

    webhook = MagicMock()
    webhook.project_key = 'PROJ'
    webhook.repo_slug = 'myrepo'
    webhook.webhook_secret = 'shared-secret'
    webhook.webhook_id = '42'
    webhook.user_id = 'kc-installer'
    webhook.last_synced = None
    mock_store.get_webhooks_by_repos = AsyncMock(
        return_value={('PROJ', 'myrepo'): webhook}
    )

    response = await get_bitbucket_dc_resources(user_id='kc-viewer')

    mock_service_cls.assert_called_once_with(external_auth_id='kc-viewer')
    mock_store.get_webhooks_by_repos.assert_awaited_once_with([('PROJ', 'myrepo')])
    assert len(response.resources) == 1
    resource = response.resources[0]
    assert resource.project_key == 'PROJ'
    assert resource.repo_slug == 'myrepo'
    assert resource.webhook_enrolled is True
    assert resource.webhook_id == '42'
    assert resource.installed_by_user_id == 'kc-installer'


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.secrets.token_urlsafe')
@patch('server.routes.integration.bitbucket_dc.webhook_store')
async def test_enroll_bitbucket_dc_webhook_generates_secret_and_stores_row(
    mock_store, mock_token_urlsafe
):
    from server.routes.integration.bitbucket_dc import (
        BitbucketDCResourceIdentifier,
        EnrollBitbucketDCWebhookRequest,
    )

    mock_token_urlsafe.return_value = 'generated-secret'
    mock_store.upsert_webhook_enrollment = AsyncMock()

    response = await enroll_bitbucket_dc_webhook(
        body=EnrollBitbucketDCWebhookRequest(
            resource=BitbucketDCResourceIdentifier(
                project_key='PROJ',
                repo_slug='myrepo',
            )
        ),
        user_id='kc-installer',
    )

    mock_store.upsert_webhook_enrollment.assert_awaited_once_with(
        project_key='PROJ',
        repo_slug='myrepo',
        user_id='kc-installer',
        webhook_secret='generated-secret',
    )
    assert response.success is True
    assert response.webhook_secret == 'generated-secret'
    assert response.webhook_url.endswith('/integration/bitbucket-dc/events')
    assert response.events == ['pr:comment:added', 'pr:comment:edited']


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.webhook_store')
async def test_update_bitbucket_dc_webhook_id_records_webhook_id(mock_store):
    from server.routes.integration.bitbucket_dc import (
        BitbucketDCResourceIdentifier,
        UpdateBitbucketDCWebhookIdRequest,
    )

    mock_store.update_webhook_id = AsyncMock(return_value=True)

    response = await update_bitbucket_dc_webhook_id(
        body=UpdateBitbucketDCWebhookIdRequest(
            resource=BitbucketDCResourceIdentifier(
                project_key='PROJ',
                repo_slug='myrepo',
            ),
            webhook_id='42',
        ),
        user_id='kc-installer',
    )

    mock_store.update_webhook_id.assert_awaited_once_with(
        project_key='PROJ',
        repo_slug='myrepo',
        webhook_id='42',
    )
    assert response.success is True
