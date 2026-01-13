"""Shared utilities for GitLab webhook installation.

This module contains reusable functions and classes for installing GitLab webhooks
that can be used by both the cron job and API routes.
"""

from typing import cast
from uuid import uuid4

from integrations.types import GitLabResourceType
from integrations.utils import GITLAB_WEBHOOK_URL
from storage.gitlab_webhook import GitlabWebhook, WebhookStatus
from storage.gitlab_webhook_store import GitlabWebhookStore

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.service_types import GitService

# Webhook configuration constants
WEBHOOK_NAME = 'OpenHands Resolver'
SCOPES: list[str] = [
    'note_events',
    'merge_requests_events',
    'confidential_issues_events',
    'issues_events',
    'confidential_note_events',
    'job_events',
    'pipeline_events',
]


class BreakLoopException(Exception):
    """Exception raised when webhook installation conditions are not met or rate limited."""

    pass


async def verify_webhook_conditions(
    gitlab_service: type[GitService],
    resource_type: GitLabResourceType,
    resource_id: str,
    webhook_store: GitlabWebhookStore,
    webhook: GitlabWebhook,
) -> None:
    """
    Verify all conditions are met for webhook installation.
    Raises BreakLoopException if any condition fails or rate limited.

    Args:
        gitlab_service: GitLab service instance
        resource_type: Type of resource (PROJECT or GROUP)
        resource_id: ID of the resource
        webhook_store: Webhook store instance
        webhook: Webhook object to verify
    """
    from integrations.gitlab.gitlab_service import SaaSGitLabService

    gitlab_service = cast(type[SaaSGitLabService], gitlab_service)

    # Check if resource exists
    does_resource_exist, status = await gitlab_service.check_resource_exists(
        resource_type, resource_id
    )

    logger.info(
        'Does resource exists',
        extra={
            'does_resource_exist': does_resource_exist,
            'status': status,
            'resource_id': resource_id,
            'resource_type': resource_type,
        },
    )

    if status == WebhookStatus.RATE_LIMITED:
        raise BreakLoopException()
    if not does_resource_exist and status != WebhookStatus.RATE_LIMITED:
        await webhook_store.delete_webhook(webhook)
        raise BreakLoopException()

    # Check if user has admin access
    (
        is_user_admin_of_resource,
        status,
    ) = await gitlab_service.check_user_has_admin_access_to_resource(
        resource_type, resource_id
    )

    logger.info(
        'Is user admin',
        extra={
            'is_user_admin': is_user_admin_of_resource,
            'status': status,
            'resource_id': resource_id,
            'resource_type': resource_type,
        },
    )

    if status == WebhookStatus.RATE_LIMITED:
        raise BreakLoopException()
    if not is_user_admin_of_resource:
        await webhook_store.delete_webhook(webhook)
        raise BreakLoopException()

    # Check if webhook already exists
    (
        does_webhook_exist_on_resource,
        status,
    ) = await gitlab_service.check_webhook_exists_on_resource(
        resource_type, resource_id, GITLAB_WEBHOOK_URL
    )

    logger.info(
        'Does webhook already exist',
        extra={
            'does_webhook_exist_on_resource': does_webhook_exist_on_resource,
            'status': status,
            'resource_id': resource_id,
            'resource_type': resource_type,
        },
    )

    if status == WebhookStatus.RATE_LIMITED:
        raise BreakLoopException()
    if does_webhook_exist_on_resource != webhook.webhook_exists:
        await webhook_store.update_webhook(
            webhook, {'webhook_exists': does_webhook_exist_on_resource}
        )

    if does_webhook_exist_on_resource:
        raise BreakLoopException()


async def install_webhook_on_resource(
    gitlab_service: type[GitService],
    resource_type: GitLabResourceType,
    resource_id: str,
    webhook_store: GitlabWebhookStore,
    webhook: GitlabWebhook,
) -> tuple[str | None, WebhookStatus | None]:
    """
    Install webhook on a GitLab resource.

    Args:
        gitlab_service: GitLab service instance
        resource_type: Type of resource (PROJECT or GROUP)
        resource_id: ID of the resource
        webhook_store: Webhook store instance
        webhook: Webhook object to install

    Returns:
        Tuple of (webhook_id, status)
    """
    from integrations.gitlab.gitlab_service import SaaSGitLabService

    gitlab_service = cast(type[SaaSGitLabService], gitlab_service)

    webhook_secret = f'{webhook.user_id}-{str(uuid4())}'
    webhook_uuid = f'{str(uuid4())}'

    webhook_id, status = await gitlab_service.install_webhook(
        resource_type=resource_type,
        resource_id=resource_id,
        webhook_name=WEBHOOK_NAME,
        webhook_url=GITLAB_WEBHOOK_URL,
        webhook_secret=webhook_secret,
        webhook_uuid=webhook_uuid,
        scopes=SCOPES,
    )

    logger.info(
        'Creating new webhook',
        extra={
            'webhook_id': webhook_id,
            'status': status,
            'resource_id': resource_id,
            'resource_type': resource_type,
        },
    )

    if status == WebhookStatus.RATE_LIMITED:
        raise BreakLoopException()

    if webhook_id:
        await webhook_store.update_webhook(
            webhook=webhook,
            update_fields={
                'webhook_secret': webhook_secret,
                'webhook_exists': True,  # webhook was created
                'webhook_url': GITLAB_WEBHOOK_URL,
                'scopes': SCOPES,
                'webhook_uuid': webhook_uuid,  # required to identify which webhook installation is sending payload
            },
        )

        logger.info(
            f'Installed webhook for {webhook.user_id} on {resource_type}:{resource_id}'
        )

    return webhook_id, status
