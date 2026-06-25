import base64
from unittest.mock import AsyncMock

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


def _azure_resource():
    return azure_devops.AzureDevOpsResourceIdentifier(
        organization='alonaking',
        project_id='project-id',
        project_name='Project',
    )


@pytest.mark.asyncio
async def test_get_azure_devops_resources_reports_installed_status(monkeypatch):
    monkeypatch.setattr(azure_devops, 'HOST_URL', 'https://app.example.com')
    monkeypatch.setattr(azure_devops, 'AZURE_DEVOPS_WEBHOOK_SECRET', 'secret')

    class FakeAzureDevOpsService:
        organization = 'alonaking'

        async def get_projects_for_webhook_setup(self):
            return [
                {
                    'organization': 'alonaking',
                    'project_id': 'project-id',
                    'project_name': 'Project',
                    'full_name': 'alonaking/Project',
                }
            ]

        async def list_service_hook_subscriptions(self):
            return [
                {
                    'id': 'pr-subscription-id',
                    'eventType': azure_devops.AZURE_DEVOPS_PR_COMMENT_EVENT,
                    'publisherInputs': {'projectId': 'project-id'},
                    'consumerInputs': {
                        'url': 'https://app.example.com/integration/azure-devops/events',
                    },
                },
                {
                    'id': 'work-item-subscription-id',
                    'eventType': azure_devops.AZURE_DEVOPS_WORK_ITEM_COMMENT_EVENT,
                    'publisherInputs': {'projectId': 'project-id'},
                    'consumerInputs': {
                        'url': 'https://app.example.com/integration/azure-devops/events',
                    },
                },
            ]

    monkeypatch.setattr(
        azure_devops,
        'SaaSAzureDevOpsService',
        lambda external_auth_id: FakeAzureDevOpsService(),
    )

    response = await azure_devops.get_azure_devops_resources(user_id='user-id')

    assert len(response.resources) == 1
    resource = response.resources[0]
    assert resource.type == 'project'
    assert resource.full_name == 'alonaking/Project'
    assert resource.webhook_installed is True
    assert resource.pr_subscription_id == 'pr-subscription-id'
    assert resource.work_item_subscription_id == 'work-item-subscription-id'
    assert (
        resource.webhook_url
        == 'https://app.example.com/integration/azure-devops/events'
    )


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
                    'publisherInputs': {'projectId': 'project-id'},
                    'consumerInputs': {
                        'url': 'https://app.example.com/integration/azure-devops/events',
                    },
                },
                {
                    'id': 'old-work-item-subscription-id',
                    'eventType': azure_devops.AZURE_DEVOPS_WORK_ITEM_COMMENT_EVENT,
                    'publisherInputs': {'projectId': 'project-id'},
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

    response = await azure_devops.reinstall_azure_devops_webhook(
        body=azure_devops.AzureDevOpsWebhookRequest(resource=_azure_resource()),
        user_id='user-id',
    )

    assert response.success is True
    assert response.pr_subscription_id == 'new-pr-subscription-id'
    assert response.work_item_subscription_id == 'new-work-item-subscription-id'
    fake_service.delete_service_hook_subscription.assert_any_await(
        'old-pr-subscription-id'
    )
    fake_service.delete_service_hook_subscription.assert_any_await(
        'old-work-item-subscription-id'
    )
    fake_service.create_pr_comment_service_hook.assert_awaited_once_with(
        project_id='project-id',
        webhook_url='https://app.example.com/integration/azure-devops/events',
        webhook_secret='secret',
    )
    fake_service.create_work_item_comment_service_hook.assert_awaited_once_with(
        project_id='project-id',
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
        await azure_devops.reinstall_azure_devops_webhook(
            body=azure_devops.AzureDevOpsWebhookRequest(resource=_azure_resource()),
            user_id='user-id',
        )

    assert exc_info.value.status_code == 503
