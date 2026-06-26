from __future__ import annotations

import base64
import hashlib
import json
import secrets
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import JSONResponse
from integrations.azure_devops.azure_devops_service import SaaSAzureDevOpsService
from integrations.models import Message, SourceType
from integrations.utils import HOST_URL, IS_LOCAL_DEPLOYMENT
from pydantic import BaseModel
from server.auth.authorization import Permission, require_permission
from server.auth.constants import AZURE_DEVOPS_WEBHOOK_SECRET
from storage.redis import get_redis_client_async

from openhands.app_server.integrations.azure_devops.service.webhooks import (
    AZURE_DEVOPS_PR_COMMENT_EVENT,
    AZURE_DEVOPS_WORK_ITEM_COMMENT_EVENT,
)
from openhands.app_server.utils.logger import openhands_logger as logger

azure_devops_integration_router = APIRouter(prefix='/integration')

_azure_devops_manager = None


def azure_devops_webhook_url() -> str:
    return f'{HOST_URL}/integration/azure-devops/events'


class AzureDevOpsWebhookStatus(BaseModel):
    organization: str
    webhook_installed: bool
    pr_webhook_installed: bool
    work_item_webhook_installed: bool
    pr_subscription_id: str | None
    work_item_subscription_id: str | None
    webhook_url: str
    webhook_secret_set: bool


class AzureDevOpsWebhookInstallationResult(BaseModel):
    organization: str
    success: bool
    error: str | None
    pr_subscription_id: str | None
    work_item_subscription_id: str | None
    webhook_url: str


def get_azure_devops_manager():
    global _azure_devops_manager
    if _azure_devops_manager is None:
        from integrations.azure_devops.azure_devops_manager import AzureDevOpsManager
        from server.auth.token_manager import TokenManager

        _azure_devops_manager = AzureDevOpsManager(TokenManager())
    return _azure_devops_manager


def _basic_auth_secret(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, encoded = authorization.partition(' ')
    if scheme.lower() != 'basic' or not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded).decode()
    except Exception:
        return None
    _, _, password = decoded.partition(':')
    return password or decoded


def _subscription_id(subscription: dict[str, Any]) -> str | None:
    subscription_id = subscription.get('id')
    return str(subscription_id) if subscription_id else None


def _subscription_consumer_url(subscription: dict[str, Any]) -> str:
    return str((subscription.get('consumerInputs') or {}).get('url') or '')


def _subscription_event_type(subscription: dict[str, Any]) -> str:
    return str(subscription.get('eventType') or '')


def _subscription_is_project_scoped(subscription: dict[str, Any]) -> bool:
    # A projectId in publisherInputs means the hook is scoped to one project,
    # not the org-wide resolver install.
    return bool((subscription.get('publisherInputs') or {}).get('projectId'))


def _subscription_is_enabled(subscription: dict[str, Any] | None) -> bool:
    # A Service Hook can exist but be disabled/suspended and not deliver events.
    return subscription is not None and subscription.get('status') == 'enabled'


def _matches_pr_comment_subscription(
    subscription: dict[str, Any],
    *,
    webhook_url: str,
) -> bool:
    # Org-wide hook = event + consumer url + no projectId (project-scoped hooks
    # at the same url must not be mistaken for the org-wide install).
    return (
        _subscription_event_type(subscription) == AZURE_DEVOPS_PR_COMMENT_EVENT
        and _subscription_consumer_url(subscription) == webhook_url
        and not _subscription_is_project_scoped(subscription)
    )


def _matches_work_item_comment_subscription(
    subscription: dict[str, Any],
    *,
    webhook_url: str,
) -> bool:
    return (
        _subscription_event_type(subscription) == AZURE_DEVOPS_WORK_ITEM_COMMENT_EVENT
        and _subscription_consumer_url(subscription) == webhook_url
        and not _subscription_is_project_scoped(subscription)
    )


# Webhook secret for local dev only; real installs set AZURE_DEVOPS_WEBHOOK_SECRET.
_LOCAL_DEPLOYMENT_WEBHOOK_SECRET = 'localdeploymentwebhooktesttoken'


def _get_webhook_secret() -> str | None:
    return (
        _LOCAL_DEPLOYMENT_WEBHOOK_SECRET
        if IS_LOCAL_DEPLOYMENT
        else AZURE_DEVOPS_WEBHOOK_SECRET
    )


def _ensure_azure_devops_webhook_secret() -> str:
    secret = _get_webhook_secret()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Azure DevOps webhook secret is not configured.',
        )
    return secret


def _ensure_azure_devops_organization(service: SaaSAzureDevOpsService) -> None:
    if not service.organization:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Azure DevOps organization is not configured.',
        )


async def verify_azure_devops_signature(
    header_webhook_secret: str | None,
    authorization: str | None,
) -> None:
    expected_secret = _get_webhook_secret()
    provided_secret = header_webhook_secret or _basic_auth_secret(authorization)
    if not expected_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Azure DevOps webhook secret is not configured.',
        )
    if not provided_secret or not secrets.compare_digest(
        provided_secret, expected_secret
    ):
        raise HTTPException(status_code=403, detail="Request signatures didn't match!")


@azure_devops_integration_router.get('/azure-devops/resources')
async def get_azure_devops_resources(
    user_id: str = Depends(require_permission(Permission.MANAGE_INTEGRATIONS)),
) -> AzureDevOpsWebhookStatus:
    """Report org-wide resolver hook installation status."""
    try:
        service = SaaSAzureDevOpsService(external_auth_id=user_id)
        _ensure_azure_devops_organization(service)
        webhook_url = azure_devops_webhook_url()

        subscriptions = await service.list_service_hook_subscriptions()
        pr_subscription = next(
            (
                subscription
                for subscription in subscriptions
                if _matches_pr_comment_subscription(
                    subscription, webhook_url=webhook_url
                )
            ),
            None,
        )
        work_item_subscription = next(
            (
                subscription
                for subscription in subscriptions
                if _matches_work_item_comment_subscription(
                    subscription, webhook_url=webhook_url
                )
            ),
            None,
        )

        # Only report installed when the hooks are enabled (a disabled/suspended
        # hook exists but won't deliver resolver events).
        pr_enabled = _subscription_is_enabled(pr_subscription)
        work_item_enabled = _subscription_is_enabled(work_item_subscription)

        return AzureDevOpsWebhookStatus(
            organization=service.organization,
            webhook_installed=pr_enabled and work_item_enabled,
            pr_webhook_installed=pr_enabled,
            work_item_webhook_installed=work_item_enabled,
            pr_subscription_id=_subscription_id(pr_subscription)
            if pr_subscription
            else None,
            work_item_subscription_id=_subscription_id(work_item_subscription)
            if work_item_subscription
            else None,
            webhook_url=webhook_url,
            webhook_secret_set=bool(IS_LOCAL_DEPLOYMENT or AZURE_DEVOPS_WEBHOOK_SECRET),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error retrieving Azure DevOps resources: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve Azure DevOps resources',
        )


@azure_devops_integration_router.post('/azure-devops/reinstall-webhook')
async def reinstall_azure_devops_webhook(
    user_id: str = Depends(require_permission(Permission.MANAGE_INTEGRATIONS)),
) -> AzureDevOpsWebhookInstallationResult:
    """Install or reinstall the org-wide Azure DevOps resolver Service Hooks."""
    service = SaaSAzureDevOpsService(external_auth_id=user_id)
    _ensure_azure_devops_organization(service)
    webhook_secret = _ensure_azure_devops_webhook_secret()
    webhook_url = azure_devops_webhook_url()

    try:
        subscriptions = await service.list_service_hook_subscriptions()
        for subscription in subscriptions:
            subscription_id = _subscription_id(subscription)
            if not subscription_id:
                continue
            if _matches_pr_comment_subscription(
                subscription, webhook_url=webhook_url
            ) or _matches_work_item_comment_subscription(
                subscription, webhook_url=webhook_url
            ):
                await service.delete_service_hook_subscription(subscription_id)

        pr_subscription = await service.create_pr_comment_service_hook(
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
        )
        work_item_subscription = await service.create_work_item_comment_service_hook(
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
        )

        pr_subscription_id = _subscription_id(pr_subscription)
        work_item_subscription_id = _subscription_id(work_item_subscription)

        logger.info(
            '[Azure DevOps] Resolver hooks installed',
            extra={
                'user_id': user_id,
                'organization': service.organization,
                'pr_subscription_id': pr_subscription_id,
                'work_item_subscription_id': work_item_subscription_id,
            },
        )

        return AzureDevOpsWebhookInstallationResult(
            organization=service.organization,
            success=True,
            error=None,
            pr_subscription_id=pr_subscription_id,
            work_item_subscription_id=work_item_subscription_id,
            webhook_url=webhook_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error installing Azure DevOps resolver hooks: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to install Azure DevOps resolver hooks',
        )


@azure_devops_integration_router.post('/azure-devops/uninstall-webhook')
async def uninstall_azure_devops_webhook(
    user_id: str = Depends(require_permission(Permission.MANAGE_INTEGRATIONS)),
) -> AzureDevOpsWebhookInstallationResult:
    """Delete the org-wide Azure DevOps resolver Service Hooks."""
    service = SaaSAzureDevOpsService(external_auth_id=user_id)
    _ensure_azure_devops_organization(service)
    webhook_url = azure_devops_webhook_url()

    try:
        subscriptions = await service.list_service_hook_subscriptions()
        deleted_pr_subscription_id: str | None = None
        deleted_work_item_subscription_id: str | None = None

        for subscription in subscriptions:
            subscription_id = _subscription_id(subscription)
            if not subscription_id:
                continue
            if _matches_pr_comment_subscription(subscription, webhook_url=webhook_url):
                await service.delete_service_hook_subscription(subscription_id)
                deleted_pr_subscription_id = subscription_id
            elif _matches_work_item_comment_subscription(
                subscription, webhook_url=webhook_url
            ):
                await service.delete_service_hook_subscription(subscription_id)
                deleted_work_item_subscription_id = subscription_id

        logger.info(
            '[Azure DevOps] Resolver hooks uninstalled',
            extra={
                'user_id': user_id,
                'organization': service.organization,
                'pr_subscription_id': deleted_pr_subscription_id,
                'work_item_subscription_id': deleted_work_item_subscription_id,
            },
        )

        return AzureDevOpsWebhookInstallationResult(
            organization=service.organization,
            success=True,
            error=None,
            pr_subscription_id=deleted_pr_subscription_id,
            work_item_subscription_id=deleted_work_item_subscription_id,
            webhook_url=webhook_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error uninstalling Azure DevOps resolver hooks: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to uninstall Azure DevOps resolver hooks',
        )


@azure_devops_integration_router.post('/azure-devops/events')
async def azure_devops_events(
    request: Request,
    background_tasks: BackgroundTasks,
    x_openhands_webhook_secret: str | None = Header(None),
    authorization: str | None = Header(None),
):
    try:
        await verify_azure_devops_signature(
            header_webhook_secret=x_openhands_webhook_secret,
            authorization=authorization,
        )

        payload_data = await request.json()
        # Dedup on the stable event id; notificationId changes on ADO retries,
        # so including it would let a retried delivery start a duplicate job.
        dedup_key = payload_data.get('id')
        if dedup_key:
            dedup_key = f'azure_devops_msg:{dedup_key}'
        else:
            dedup_json = json.dumps(payload_data, sort_keys=True)
            dedup_hash = hashlib.sha256(dedup_json.encode()).hexdigest()
            dedup_key = f'azure_devops_msg:{dedup_hash}'

        redis = get_redis_client_async()
        created = await redis.set(dedup_key, 1, nx=True, ex=60)
        if not created:
            logger.info('azure_devops_is_duplicate')
            return JSONResponse(
                status_code=200,
                content={'message': 'Duplicate Azure DevOps event ignored.'},
            )

        message = Message(
            source=SourceType.AZURE_DEVOPS,
            message={
                'payload': payload_data,
                'event_key': payload_data.get('eventType'),
            },
        )
        # Process in the background so we return 200 fast; conversation/sandbox
        # startup can exceed Service Hooks' delivery timeout -> retries. Mirrors
        # the Jira DC / Bitbucket DC routes. Signature is verified above.
        background_tasks.add_task(get_azure_devops_manager().receive_message, message)

        return JSONResponse(
            status_code=200,
            content={'message': 'Azure DevOps events endpoint reached successfully.'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error processing Azure DevOps event: {e}')
        return JSONResponse(status_code=400, content={'error': 'Invalid payload.'})
