import asyncio
import hashlib
import json

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from integrations.gitlab.gitlab_manager import GitlabManager
from integrations.gitlab.gitlab_service import SaaSGitLabService
from integrations.gitlab.webhook_installation import (
    BreakLoopException,
    install_webhook_on_resource,
    verify_webhook_conditions,
)
from integrations.models import Message, SourceType
from integrations.types import GitLabResourceType
from integrations.utils import GITLAB_WEBHOOK_URL
from pydantic import BaseModel
from server.auth.token_manager import TokenManager
from storage.gitlab_webhook import GitlabWebhook
from storage.gitlab_webhook_store import GitlabWebhookStore

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.gitlab.gitlab_service import GitLabServiceImpl
from openhands.server.shared import sio
from openhands.server.user_auth import get_user_id

gitlab_integration_router = APIRouter(prefix='/integration')
webhook_store = GitlabWebhookStore()

token_manager = TokenManager()
gitlab_manager = GitlabManager(token_manager)


# Request/Response models
class ResourceIdentifier(BaseModel):
    type: GitLabResourceType
    id: str


class ReinstallWebhookRequest(BaseModel):
    resource: ResourceIdentifier


class ResourceWithWebhookStatus(BaseModel):
    id: str
    name: str
    full_path: str
    type: str
    webhook_installed: bool
    webhook_uuid: str | None
    last_synced: str | None


class GitLabResourcesResponse(BaseModel):
    resources: list[ResourceWithWebhookStatus]


class ResourceInstallationResult(BaseModel):
    resource_id: str
    resource_type: str
    success: bool
    error: str | None


async def verify_gitlab_signature(
    header_webhook_secret: str, webhook_uuid: str, user_id: str
):
    if not header_webhook_secret or not webhook_uuid or not user_id:
        raise HTTPException(status_code=403, detail='Required payload headers missing!')

    webhook_secret = await webhook_store.get_webhook_secret(
        webhook_uuid=webhook_uuid, user_id=user_id
    )

    if header_webhook_secret != webhook_secret:
        raise HTTPException(status_code=403, detail="Request signatures didn't match!")


@gitlab_integration_router.post('/gitlab/events')
async def gitlab_events(
    request: Request,
    x_gitlab_token: str = Header(None),
    x_openhands_webhook_id: str = Header(None),
    x_openhands_user_id: str = Header(None),
):
    try:
        await verify_gitlab_signature(
            header_webhook_secret=x_gitlab_token,
            webhook_uuid=x_openhands_webhook_id,
            user_id=x_openhands_user_id,
        )

        payload_data = await request.json()
        object_attributes = payload_data.get('object_attributes', {})
        dedup_key = object_attributes.get('id')

        if not dedup_key:
            # Hash entire payload if payload doesn't contain payload ID
            dedup_json = json.dumps(payload_data, sort_keys=True)
            dedup_hash = hashlib.sha256(dedup_json.encode()).hexdigest()
            dedup_key = f'gitlab_msg: {dedup_hash}'

        redis = sio.manager.redis
        created = await redis.set(dedup_key, 1, nx=True, ex=60)
        if not created:
            logger.info('gitlab_is_duplicate')
            return JSONResponse(
                status_code=200,
                content={'message': 'Duplicate GitLab event ignored.'},
            )

        message = Message(
            source=SourceType.GITLAB,
            message={
                'payload': payload_data,
                'installation_id': x_openhands_webhook_id,
            },
        )

        await gitlab_manager.receive_message(message)

        return JSONResponse(
            status_code=200,
            content={'message': 'GitLab events endpoint reached successfully.'},
        )

    except Exception as e:
        logger.exception(f'Error processing GitLab event: {e}')
        return JSONResponse(status_code=400, content={'error': 'Invalid payload.'})


@gitlab_integration_router.get('/gitlab/resources')
async def get_gitlab_resources(
    user_id: str = Depends(get_user_id),
) -> GitLabResourcesResponse:
    """Get all GitLab projects and groups where the user has admin access.

    Returns a list of resources with their webhook installation status.
    """
    try:
        # Get GitLab service for the user
        gitlab_service = GitLabServiceImpl(external_auth_id=user_id)

        if not isinstance(gitlab_service, SaaSGitLabService):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Only SaaS GitLab service is supported',
            )

        # Fetch projects and groups with admin access
        projects, groups = await gitlab_service.get_user_resources_with_admin_access()

        # Filter out projects that belong to a group (nested projects)
        # We only want top-level personal projects since group webhooks cover nested projects
        filtered_projects = [
            project
            for project in projects
            if project.get('namespace', {}).get('kind') != 'group'
        ]

        # Extract IDs for bulk fetching
        project_ids = [str(project['id']) for project in filtered_projects]
        group_ids = [str(group['id']) for group in groups]

        # Bulk fetch webhook records from database (organization-wide)
        (
            project_webhook_map,
            group_webhook_map,
        ) = await webhook_store.get_webhooks_by_resources(project_ids, group_ids)

        # Parallelize GitLab API calls to check webhook status for all resources
        async def check_project_webhook(project):
            project_id = str(project['id'])
            webhook_exists, _ = await gitlab_service.check_webhook_exists_on_resource(
                GitLabResourceType.PROJECT, project_id, GITLAB_WEBHOOK_URL
            )
            return project_id, webhook_exists

        async def check_group_webhook(group):
            group_id = str(group['id'])
            webhook_exists, _ = await gitlab_service.check_webhook_exists_on_resource(
                GitLabResourceType.GROUP, group_id, GITLAB_WEBHOOK_URL
            )
            return group_id, webhook_exists

        # Gather all API calls in parallel
        project_checks = [
            check_project_webhook(project) for project in filtered_projects
        ]
        group_checks = [check_group_webhook(group) for group in groups]

        # Execute all checks concurrently
        all_results = await asyncio.gather(*(project_checks + group_checks))

        # Split results back into projects and groups
        num_projects = len(filtered_projects)
        project_results = all_results[:num_projects]
        group_results = all_results[num_projects:]

        # Build response
        resources = []

        # Add projects with their webhook status
        for project, (project_id, webhook_exists) in zip(
            filtered_projects, project_results
        ):
            webhook = project_webhook_map.get(project_id)

            resources.append(
                ResourceWithWebhookStatus(
                    id=project_id,
                    name=project.get('name', ''),
                    full_path=project.get('path_with_namespace', ''),
                    type='project',
                    webhook_installed=webhook_exists,
                    webhook_uuid=webhook.webhook_uuid if webhook else None,
                    last_synced=(
                        webhook.last_synced.isoformat()
                        if webhook and webhook.last_synced
                        else None
                    ),
                )
            )

        # Add groups with their webhook status
        for group, (group_id, webhook_exists) in zip(groups, group_results):
            webhook = group_webhook_map.get(group_id)

            resources.append(
                ResourceWithWebhookStatus(
                    id=group_id,
                    name=group.get('name', ''),
                    full_path=group.get('full_path', ''),
                    type='group',
                    webhook_installed=webhook_exists,
                    webhook_uuid=webhook.webhook_uuid if webhook else None,
                    last_synced=(
                        webhook.last_synced.isoformat()
                        if webhook and webhook.last_synced
                        else None
                    ),
                )
            )

        logger.info(
            'Retrieved GitLab resources',
            extra={
                'user_id': user_id,
                'project_count': len(projects),
                'group_count': len(groups),
            },
        )

        return GitLabResourcesResponse(resources=resources)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error retrieving GitLab resources: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve GitLab resources',
        )


@gitlab_integration_router.post('/gitlab/reinstall-webhook')
async def reinstall_gitlab_webhook(
    body: ReinstallWebhookRequest,
    user_id: str = Depends(get_user_id),
) -> ResourceInstallationResult:
    """Reinstall GitLab webhook for a specific resource immediately.

    This endpoint validates permissions, resets webhook status in the database,
    and immediately installs the webhook on the specified resource.
    """
    try:
        # Get GitLab service for the user
        gitlab_service = GitLabServiceImpl(external_auth_id=user_id)

        if not isinstance(gitlab_service, SaaSGitLabService):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Only SaaS GitLab service is supported',
            )

        resource_id = body.resource.id
        resource_type = body.resource.type

        # Check if user has admin access to this resource
        (
            has_admin_access,
            check_status,
        ) = await gitlab_service.check_user_has_admin_access_to_resource(
            resource_type, resource_id
        )

        if not has_admin_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='User does not have admin access to this resource',
            )

        # Reset webhook in database (organization-wide, not user-specific)
        # This allows any admin user to reinstall webhooks
        await webhook_store.reset_webhook_for_reinstallation_by_resource(
            resource_type, resource_id, user_id
        )

        # Get or create webhook record (without user_id filter)
        webhook = await webhook_store.get_webhook_by_resource_only(
            resource_type, resource_id
        )

        if not webhook:
            # Create new webhook record
            webhook = GitlabWebhook(
                user_id=user_id,  # Track who created it
                project_id=resource_id
                if resource_type == GitLabResourceType.PROJECT
                else None,
                group_id=resource_id
                if resource_type == GitLabResourceType.GROUP
                else None,
                webhook_exists=False,
            )
            await webhook_store.store_webhooks([webhook])
            # Fetch it again to get the ID (without user_id filter)
            webhook = await webhook_store.get_webhook_by_resource_only(
                resource_type, resource_id
            )

        # Verify conditions and install webhook
        try:
            await verify_webhook_conditions(
                gitlab_service=gitlab_service,
                resource_type=resource_type,
                resource_id=resource_id,
                webhook_store=webhook_store,
                webhook=webhook,
            )

            # Install the webhook
            webhook_id, install_status = await install_webhook_on_resource(
                gitlab_service=gitlab_service,
                resource_type=resource_type,
                resource_id=resource_id,
                webhook_store=webhook_store,
                webhook=webhook,
            )

            if webhook_id:
                logger.info(
                    'GitLab webhook reinstalled successfully',
                    extra={
                        'user_id': user_id,
                        'resource_type': resource_type.value,
                        'resource_id': resource_id,
                    },
                )
                return ResourceInstallationResult(
                    resource_id=resource_id,
                    resource_type=resource_type.value,
                    success=True,
                    error=None,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail='Failed to install webhook',
                )

        except BreakLoopException:
            # Conditions not met or webhook already exists
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Webhook installation conditions not met or webhook already exists',
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error reinstalling GitLab webhook: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to reinstall webhook',
        )
