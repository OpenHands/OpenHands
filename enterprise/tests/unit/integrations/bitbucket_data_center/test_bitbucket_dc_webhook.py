"""Tests for the Bitbucket Data Center webhook route."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from server.routes.integration.bitbucket_dc import (
    BITBUCKET_DC_WEBHOOK_EVENTS,
    bitbucket_dc_connection_events,
    enroll_bitbucket_dc_webhook,
    get_bitbucket_dc_resources,
    reinstall_bitbucket_dc_webhook,
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


def _webhook(
    *,
    webhook_id: int = 123,
    project_key: str = 'PROJ',
    repo_slug: str = 'myrepo',
    webhook_secret: str = 'shared-secret',
    user_id: str = 'kc-installer',
) -> MagicMock:
    webhook = MagicMock()
    webhook.id = webhook_id
    webhook.project_key = project_key
    webhook.repo_slug = repo_slug
    webhook.webhook_secret = webhook_secret
    webhook.user_id = user_id
    return webhook


def _scheduled_message(background_tasks, mock_manager):
    """Return the Message the route scheduled for manager.receive_message.

    The route returns 200 fast and backgrounds the manager, so the dispatch is
    a scheduled task arg rather than an awaited call.
    """
    for call in background_tasks.add_task.call_args_list:
        if call.args and call.args[0] is mock_manager.receive_message:
            return call.args[1]
    raise AssertionError('receive_message was not scheduled as a background task')


def test_bitbucket_dc_webhook_events_cover_automation_sources():
    assert {
        'repo:refs_changed',
        'repo:comment:added',
        'repo:comment:edited',
        'repo:comment:deleted',
        'pr:opened',
        'pr:from_ref_updated',
        'pr:modified',
        'pr:reviewer:approved',
        'pr:reviewer:unapproved',
        'pr:reviewer:needs_work',
        'pr:merged',
        'pr:declined',
        'pr:deleted',
        'pr:comment:added',
        'pr:comment:edited',
        'pr:comment:deleted',
    }.issubset(set(BITBUCKET_DC_WEBHOOK_EVENTS))


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
@patch('server.routes.integration.bitbucket_dc.get_redis_client_async')
async def test_signature_verification_rejects_bad_signature_with_403(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_by_id = AsyncMock(return_value=_webhook())
    mock_manager.receive_message = AsyncMock()
    mock_get_redis_client_async.return_value = AsyncMock()

    body = _pr_comment_body()

    with pytest.raises(HTTPException) as exc:
        await bitbucket_dc_connection_events(
            connection_id=123,
            request=_request_with_body(body),
            background_tasks=MagicMock(),
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
async def test_connection_event_uses_connection_repository_when_payload_identity_missing(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_by_id = AsyncMock(return_value=_webhook())
    mock_manager.receive_message = AsyncMock()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    mock_get_redis_client_async.return_value = redis

    body = json.dumps(
        {
            'pullRequest': {'id': 1},
            'comment': {'id': 99, 'text': 'Hey @openhands'},
        }
    ).encode()

    background_tasks = MagicMock()
    response = await bitbucket_dc_connection_events(
        connection_id=123,
        request=_request_with_body(body),
        background_tasks=background_tasks,
        x_hub_signature=_signed(body),
        x_event_key='pr:comment:added',
        x_request_id='req-1',
    )

    dispatched = _scheduled_message(background_tasks, mock_manager)
    assert dispatched.message['installation_id'] == 'PROJ/myrepo'
    assert response.status_code == 200


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
@patch('server.routes.integration.bitbucket_dc.get_redis_client_async')
async def test_duplicate_event_returns_200_and_skips_dispatch(
    mock_get_redis_client_async, mock_manager, mock_store
):
    mock_store.get_webhook_by_id = AsyncMock(return_value=_webhook())
    mock_manager.receive_message = AsyncMock()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=False)  # duplicate
    mock_get_redis_client_async.return_value = redis

    body = _pr_comment_body()

    response = await bitbucket_dc_connection_events(
        connection_id=123,
        request=_request_with_body(body),
        background_tasks=MagicMock(),
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
    mock_store.get_webhook_by_id = AsyncMock(return_value=_webhook())
    mock_manager.receive_message = AsyncMock()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    mock_get_redis_client_async.return_value = redis

    body = _pr_comment_body()

    background_tasks = MagicMock()
    response = await bitbucket_dc_connection_events(
        connection_id=123,
        request=_request_with_body(body),
        background_tasks=background_tasks,
        x_hub_signature=_signed(body),
        x_event_key='pr:comment:added',
        x_request_id='req-1',
    )

    dispatched = _scheduled_message(background_tasks, mock_manager)
    assert dispatched.source.value == 'bitbucket_data_center'
    assert dispatched.message['event_key'] == 'pr:comment:added'
    assert dispatched.message['installation_id'] == 'PROJ/myrepo'
    assert response.status_code == 200


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.automation_event_service')
@patch('server.routes.integration.bitbucket_dc.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
@patch('server.routes.integration.bitbucket_dc.get_redis_client_async')
async def test_valid_event_forwards_to_automations_when_enabled(
    mock_get_redis_client_async,
    mock_manager,
    mock_store,
    mock_automation_service,
):
    mock_store.get_webhook_by_id = AsyncMock(return_value=_webhook())
    mock_manager.receive_message = AsyncMock()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    mock_get_redis_client_async.return_value = redis

    body = _pr_comment_body()
    background_tasks = MagicMock()

    with patch(
        'server.routes.integration.bitbucket_dc.AUTOMATION_EVENT_FORWARDING_ENABLED',
        True,
    ):
        response = await bitbucket_dc_connection_events(
            connection_id=123,
            request=_request_with_body(body),
            background_tasks=background_tasks,
            x_hub_signature=_signed(body),
            x_event_key='pr:opened',
            x_request_id='req-1',
        )

    assert response.status_code == 200
    # Both the automation forward and the manager dispatch are scheduled.
    background_tasks.add_task.assert_any_call(
        mock_automation_service.forward_event,
        provider=ProviderType.BITBUCKET_DATA_CENTER,
        payload={
            **json.loads(body),
            'eventKey': 'pr:opened',
        },
        installation_id='PROJ/myrepo',
    )
    assert _scheduled_message(background_tasks, mock_manager) is not None


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
async def test_diagnostics_ping_returns_200_without_dispatch(mock_manager):
    mock_manager.receive_message = AsyncMock()

    response = await bitbucket_dc_connection_events(
        connection_id=123,
        request=_request_with_body(b'{}'),
        background_tasks=MagicMock(),
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
    webhook.id = 7
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
    assert resource.connection_id == 7
    assert resource.webhook_enrolled is True
    assert resource.webhook_id == '42'
    assert resource.webhook_url.endswith(
        '/integration/bitbucket-dc/connections/7/events'
    )
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
    webhook = MagicMock()
    webhook.id = 123
    mock_store.upsert_webhook_enrollment = AsyncMock(return_value=webhook)

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
    assert response.connection_id == 123
    assert response.webhook_secret == 'generated-secret'
    assert response.webhook_url.endswith(
        '/integration/bitbucket-dc/connections/123/events'
    )
    assert response.events == BITBUCKET_DC_WEBHOOK_EVENTS


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.secrets.token_urlsafe')
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.SaaSBitbucketDCService')
async def test_reinstall_bitbucket_dc_webhook_installs_connection_scoped_url(
    mock_service_cls, mock_store, mock_token_urlsafe
):
    from server.routes.integration.bitbucket_dc import (
        BitbucketDCResourceIdentifier,
        BitbucketDCWebhookRequest,
    )

    mock_token_urlsafe.return_value = 'generated-secret'
    webhook = MagicMock()
    webhook.id = 123
    mock_store.ensure_webhook_enrollment = AsyncMock(return_value=webhook)
    mock_store.upsert_webhook_enrollment = AsyncMock()

    service = MagicMock()
    service.user_has_admin_access = AsyncMock(return_value=True)
    service.check_webhook_exists_on_repository = AsyncMock(return_value=(False, None))
    service.create_repository_webhook = AsyncMock(return_value='101')
    mock_service_cls.return_value = service

    response = await reinstall_bitbucket_dc_webhook(
        body=BitbucketDCWebhookRequest(
            resource=BitbucketDCResourceIdentifier(
                project_key='PROJ',
                repo_slug='myrepo',
            )
        ),
        user_id='kc-installer',
    )

    mock_store.ensure_webhook_enrollment.assert_awaited_once_with(
        project_key='PROJ',
        repo_slug='myrepo',
        user_id='kc-installer',
    )
    service.create_repository_webhook.assert_awaited_once()
    service.check_webhook_exists_on_repository.assert_awaited_once_with(
        'PROJ',
        'myrepo',
        'https://app.all-hands.dev/integration/bitbucket-dc/connections/123/events',
    )
    create_kwargs = service.create_repository_webhook.await_args.kwargs
    assert create_kwargs['webhook_url'].endswith(
        '/integration/bitbucket-dc/connections/123/events'
    )
    mock_store.upsert_webhook_enrollment.assert_awaited_once_with(
        project_key='PROJ',
        repo_slug='myrepo',
        user_id='kc-installer',
        webhook_id='101',
        webhook_secret='generated-secret',
    )
    assert response.success is True
    assert response.webhook_id == '101'
    assert response.connection_id == 123
    assert response.webhook_url.endswith(
        '/integration/bitbucket-dc/connections/123/events'
    )


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
@patch('server.routes.integration.bitbucket_dc.get_redis_client_async')
async def test_connection_event_verifies_with_connection_secret_and_dispatches_installer(
    mock_get_redis_client_async, mock_manager, mock_store
):
    webhook = MagicMock()
    webhook.id = 123
    webhook.project_key = 'PROJ'
    webhook.repo_slug = 'myrepo'
    webhook.webhook_secret = 'shared-secret'
    webhook.user_id = 'kc-installer'
    mock_store.get_webhook_by_id = AsyncMock(return_value=webhook)
    mock_manager.receive_message = AsyncMock()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    mock_get_redis_client_async.return_value = redis

    body = _pr_comment_body()

    background_tasks = MagicMock()
    response = await bitbucket_dc_connection_events(
        connection_id=123,
        request=_request_with_body(body),
        background_tasks=background_tasks,
        x_hub_signature=_signed(body),
        x_event_key='pr:comment:added',
        x_request_id='req-1',
    )

    mock_store.get_webhook_by_id.assert_awaited_once_with(123)
    dispatched = _scheduled_message(background_tasks, mock_manager)
    assert dispatched.message['connection_id'] == 123
    assert dispatched.message['installer_user_id'] == 'kc-installer'
    assert dispatched.message['installation_id'] == 'PROJ/myrepo'
    assert response.status_code == 200


@pytest.mark.asyncio
@patch('server.routes.integration.bitbucket_dc.IS_LOCAL_DEPLOYMENT', False)
@patch('server.routes.integration.bitbucket_dc.webhook_store')
@patch('server.routes.integration.bitbucket_dc.bitbucket_dc_manager')
@patch('server.routes.integration.bitbucket_dc.get_redis_client_async')
async def test_connection_event_rejects_payload_for_different_repository(
    mock_get_redis_client_async, mock_manager, mock_store
):
    webhook = MagicMock()
    webhook.id = 123
    webhook.project_key = 'OTHER'
    webhook.repo_slug = 'repo'
    webhook.webhook_secret = 'shared-secret'
    webhook.user_id = 'kc-installer'
    mock_store.get_webhook_by_id = AsyncMock(return_value=webhook)
    mock_manager.receive_message = AsyncMock()
    mock_get_redis_client_async.return_value = AsyncMock()

    body = _pr_comment_body()

    with pytest.raises(HTTPException) as exc:
        await bitbucket_dc_connection_events(
            connection_id=123,
            request=_request_with_body(body),
            background_tasks=MagicMock(),
            x_hub_signature=_signed(body),
            x_event_key='pr:comment:added',
            x_request_id='req-1',
        )

    assert exc.value.status_code == 403
    mock_manager.receive_message.assert_not_called()


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
