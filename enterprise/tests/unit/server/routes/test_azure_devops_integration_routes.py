import base64
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from server.routes.integration import azure_devops


@pytest.mark.asyncio
async def test_verify_azure_devops_signature_accepts_header_secret(monkeypatch):
    monkeypatch.setattr(azure_devops, 'IS_LOCAL_DEPLOYMENT', False)
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'expected')

    await azure_devops.verify_azure_devops_signature('expected', None)


@pytest.mark.asyncio
async def test_verify_azure_devops_signature_accepts_basic_auth(monkeypatch):
    monkeypatch.setattr(azure_devops, 'IS_LOCAL_DEPLOYMENT', False)
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'expected')
    encoded = base64.b64encode(b'openhands:expected').decode()

    await azure_devops.verify_azure_devops_signature(None, f'Basic {encoded}')


@pytest.mark.asyncio
async def test_verify_azure_devops_signature_rejects_bad_secret(monkeypatch):
    monkeypatch.setattr(azure_devops, 'IS_LOCAL_DEPLOYMENT', False)
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'expected')

    with pytest.raises(HTTPException) as exc_info:
        await azure_devops.verify_azure_devops_signature('wrong', None)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_get_azure_devops_resources_reports_installed_status(monkeypatch):
    monkeypatch.setattr(azure_devops, 'HOST_URL', 'https://app.example.com')
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'secret')

    class FakeAzureDevOpsService:
        organization = 'alonaking'

        async def list_service_hook_subscriptions(self):
            # Org-wide subscriptions carry no projectId.
            return [
                {
                    'id': 'pr-subscription-id',
                    'eventType': azure_devops.AZURE_DEVOPS_PR_COMMENT_EVENT,
                    'publisherInputs': {},
                    'consumerInputs': {
                        'url': 'https://app.example.com/integration/azure-devops/events',
                    },
                    'status': 'enabled',
                },
                {
                    'id': 'work-item-subscription-id',
                    'eventType': azure_devops.AZURE_DEVOPS_WORK_ITEM_COMMENT_EVENT,
                    'publisherInputs': {},
                    'consumerInputs': {
                        'url': 'https://app.example.com/integration/azure-devops/events',
                    },
                    'status': 'enabled',
                },
            ]

    monkeypatch.setattr(
        azure_devops,
        'SaaSAzureDevOpsService',
        lambda external_auth_id: FakeAzureDevOpsService(),
    )

    status = await azure_devops.get_azure_devops_resources(user_id='user-id')

    assert status.organization == 'alonaking'
    assert status.webhook_installed is True
    assert status.pr_webhook_installed is True
    assert status.work_item_webhook_installed is True
    assert status.pr_subscription_id == 'pr-subscription-id'
    assert status.work_item_subscription_id == 'work-item-subscription-id'
    assert (
        status.webhook_url == 'https://app.example.com/integration/azure-devops/events'
    )


@pytest.mark.asyncio
async def test_get_azure_devops_resources_reports_disabled_hook_as_not_installed(
    monkeypatch,
):
    monkeypatch.setattr(azure_devops, 'HOST_URL', 'https://app.example.com')
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'secret')
    url = 'https://app.example.com/integration/azure-devops/events'

    class FakeAzureDevOpsService:
        organization = 'alonaking'

        async def list_service_hook_subscriptions(self):
            return [
                {
                    'id': 'pr-subscription-id',
                    'eventType': azure_devops.AZURE_DEVOPS_PR_COMMENT_EVENT,
                    'publisherInputs': {},
                    'consumerInputs': {'url': url},
                    'status': 'enabled',
                },
                {
                    'id': 'work-item-subscription-id',
                    'eventType': azure_devops.AZURE_DEVOPS_WORK_ITEM_COMMENT_EVENT,
                    'publisherInputs': {},
                    'consumerInputs': {'url': url},
                    'status': 'disabledByUser',
                },
            ]

    monkeypatch.setattr(
        azure_devops,
        'SaaSAzureDevOpsService',
        lambda external_auth_id: FakeAzureDevOpsService(),
    )

    status = await azure_devops.get_azure_devops_resources(user_id='user-id')

    # Disabled work-item hook -> org not fully installed, but id still surfaced.
    assert status.webhook_installed is False
    assert status.pr_webhook_installed is True
    assert status.work_item_webhook_installed is False
    assert status.work_item_subscription_id == 'work-item-subscription-id'


def test_matchers_require_absent_project_id():
    url = 'https://app.example.com/integration/azure-devops/events'
    org_wide = {
        'eventType': azure_devops.AZURE_DEVOPS_PR_COMMENT_EVENT,
        'publisherInputs': {},
        'consumerInputs': {'url': url},
    }
    project_scoped = {
        'eventType': azure_devops.AZURE_DEVOPS_PR_COMMENT_EVENT,
        'publisherInputs': {'projectId': 'proj-guid'},
        'consumerInputs': {'url': url},
    }

    assert (
        azure_devops._matches_pr_comment_subscription(org_wide, webhook_url=url) is True
    )
    assert (
        azure_devops._matches_pr_comment_subscription(project_scoped, webhook_url=url)
        is False
    )


@pytest.mark.asyncio
async def test_get_azure_devops_resources_ignores_project_scoped_hooks(monkeypatch):
    monkeypatch.setattr(azure_devops, 'HOST_URL', 'https://app.example.com')
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'secret')

    url = 'https://app.example.com/integration/azure-devops/events'

    class FakeAzureDevOpsService:
        organization = 'alonaking'

        async def list_service_hook_subscriptions(self):
            # Same event types + url, but project-scoped (projectId set) -> must
            # NOT be reported as the org-wide install.
            return [
                {
                    'id': 'pr-project-scoped',
                    'eventType': azure_devops.AZURE_DEVOPS_PR_COMMENT_EVENT,
                    'publisherInputs': {'projectId': 'proj-guid'},
                    'consumerInputs': {'url': url},
                },
                {
                    'id': 'wi-project-scoped',
                    'eventType': azure_devops.AZURE_DEVOPS_WORK_ITEM_COMMENT_EVENT,
                    'publisherInputs': {'projectId': 'proj-guid'},
                    'consumerInputs': {'url': url},
                },
            ]

    monkeypatch.setattr(
        azure_devops,
        'SaaSAzureDevOpsService',
        lambda external_auth_id: FakeAzureDevOpsService(),
    )

    status = await azure_devops.get_azure_devops_resources(user_id='user-id')

    assert status.webhook_installed is False
    assert status.pr_webhook_installed is False
    assert status.work_item_webhook_installed is False
    assert status.pr_subscription_id is None
    assert status.work_item_subscription_id is None


@pytest.mark.asyncio
async def test_reinstall_azure_devops_webhook_replaces_existing_hooks(monkeypatch):
    monkeypatch.setattr(azure_devops, 'HOST_URL', 'https://app.example.com')
    monkeypatch.setattr(azure_devops, 'IS_LOCAL_DEPLOYMENT', False)
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'secret')

    class FakeAzureDevOpsService:
        organization = 'alonaking'

        def __init__(self):
            self.delete_service_hook_subscription = AsyncMock()
            self.create_pr_comment_service_hook = AsyncMock(
                return_value={'id': 'new-pr-subscription-id'}
            )
            self.create_work_item_comment_service_hook = AsyncMock(
                return_value={'id': 'new-work-item-subscription-id'}
            )

        async def list_service_hook_subscriptions(self):
            return [
                {
                    'id': 'old-pr-subscription-id',
                    'eventType': azure_devops.AZURE_DEVOPS_PR_COMMENT_EVENT,
                    'publisherInputs': {},
                    'consumerInputs': {
                        'url': 'https://app.example.com/integration/azure-devops/events',
                    },
                },
                {
                    'id': 'old-work-item-subscription-id',
                    'eventType': azure_devops.AZURE_DEVOPS_WORK_ITEM_COMMENT_EVENT,
                    'publisherInputs': {},
                    'consumerInputs': {
                        'url': 'https://app.example.com/integration/azure-devops/events',
                    },
                },
            ]

    fake_service = FakeAzureDevOpsService()
    monkeypatch.setattr(
        azure_devops,
        'SaaSAzureDevOpsService',
        lambda external_auth_id: fake_service,
    )

    response = await azure_devops.reinstall_azure_devops_webhook(user_id='user-id')

    assert response.success is True
    assert response.organization == 'alonaking'
    assert response.pr_subscription_id == 'new-pr-subscription-id'
    assert response.work_item_subscription_id == 'new-work-item-subscription-id'
    fake_service.delete_service_hook_subscription.assert_any_await(
        'old-pr-subscription-id'
    )
    fake_service.delete_service_hook_subscription.assert_any_await(
        'old-work-item-subscription-id'
    )
    fake_service.create_pr_comment_service_hook.assert_awaited_once_with(
        webhook_url='https://app.example.com/integration/azure-devops/events',
        webhook_secret='secret',
    )
    fake_service.create_work_item_comment_service_hook.assert_awaited_once_with(
        webhook_url='https://app.example.com/integration/azure-devops/events',
        webhook_secret='secret',
    )


@pytest.mark.asyncio
async def test_reinstall_azure_devops_webhook_requires_configured_secret(monkeypatch):
    monkeypatch.setattr(azure_devops, 'IS_LOCAL_DEPLOYMENT', False)
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', '')

    class FakeAzureDevOpsService:
        organization = 'alonaking'

    monkeypatch.setattr(
        azure_devops,
        'SaaSAzureDevOpsService',
        lambda external_auth_id: FakeAzureDevOpsService(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await azure_devops.reinstall_azure_devops_webhook(user_id='user-id')

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_events_route_schedules_receive_message_in_background(monkeypatch):
    monkeypatch.setattr(azure_devops, 'IS_LOCAL_DEPLOYMENT', False)
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'secret')

    fake_redis = MagicMock()
    fake_redis.set = AsyncMock(return_value=True)
    monkeypatch.setattr(azure_devops, 'get_redis_client_async', lambda: fake_redis)

    fake_manager = MagicMock()
    fake_manager.receive_message = AsyncMock()
    monkeypatch.setattr(azure_devops, 'get_azure_devops_manager', lambda: fake_manager)

    class _FakeRequest:
        async def json(self):
            return {'id': 'evt-1', 'eventType': 'workitem.commented'}

    background_tasks = MagicMock()

    response = await azure_devops.azure_devops_events(
        request=_FakeRequest(),
        background_tasks=background_tasks,
        x_openhands_webhook_secret='secret',
        authorization=None,
    )

    assert response.status_code == 200
    # receive_message is queued, not awaited inline (avoids Service Hooks timeout).
    background_tasks.add_task.assert_called_once_with(fake_manager.receive_message, ANY)
    fake_manager.receive_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_events_route_skips_duplicate_without_scheduling(monkeypatch):
    monkeypatch.setattr(azure_devops, 'IS_LOCAL_DEPLOYMENT', False)
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'secret')

    fake_redis = MagicMock()
    fake_redis.set = AsyncMock(return_value=False)  # already seen -> duplicate
    monkeypatch.setattr(azure_devops, 'get_redis_client_async', lambda: fake_redis)

    fake_manager = MagicMock()
    fake_manager.receive_message = AsyncMock()
    monkeypatch.setattr(azure_devops, 'get_azure_devops_manager', lambda: fake_manager)

    class _FakeRequest:
        async def json(self):
            return {'id': 'evt-1', 'eventType': 'workitem.commented'}

    background_tasks = MagicMock()

    response = await azure_devops.azure_devops_events(
        request=_FakeRequest(),
        background_tasks=background_tasks,
        x_openhands_webhook_secret='secret',
        authorization=None,
    )

    assert response.status_code == 200
    background_tasks.add_task.assert_not_called()
