from __future__ import annotations

import hashlib
import hmac
import json
import secrets

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
from integrations.bitbucket_data_center.bitbucket_dc_manager import BitbucketDCManager
from integrations.bitbucket_data_center.bitbucket_dc_service import (
    SaaSBitbucketDCService,
)
from integrations.models import Message, SourceType
from integrations.utils import HOST_URL, IS_LOCAL_DEPLOYMENT
from pydantic import BaseModel
from server.auth.authorization import Permission, require_permission
from server.auth.constants import AUTOMATION_EVENT_FORWARDING_ENABLED
from server.auth.token_manager import TokenManager
from server.services.automation_event_service import AutomationEventService
from storage.bitbucket_dc_webhook_store import BitbucketDCWebhookStore
from storage.redis import get_redis_client_async

from openhands.app_server.config_api.config_models import AppMode
from openhands.app_server.integrations.provider import ProviderType
from openhands.app_server.utils.logger import openhands_logger as logger

bitbucket_dc_integration_router = APIRouter(prefix='/integration')

webhook_store = BitbucketDCWebhookStore()
token_manager = TokenManager()
bitbucket_dc_manager = BitbucketDCManager(token_manager)
automation_event_service = AutomationEventService(token_manager)

BITBUCKET_DC_WEBHOOK_NAME = 'OpenHands Resolver'
BITBUCKET_DC_WEBHOOK_EVENTS = [
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
]
BITBUCKET_DC_WEBHOOK_URL = f'{HOST_URL}/integration/bitbucket-dc/events'


class BitbucketDCResourceIdentifier(BaseModel):
    project_key: str
    repo_slug: str


class BitbucketDCResourceWithWebhookStatus(BaseModel):
    project_key: str
    repo_slug: str
    name: str
    full_name: str
    type: str = 'repository'
    webhook_enrolled: bool
    webhook_id: str | None
    webhook_secret_set: bool
    installed_by_user_id: str | None
    last_synced: str | None


class BitbucketDCResourcesResponse(BaseModel):
    resources: list[BitbucketDCResourceWithWebhookStatus]


class EnrollBitbucketDCWebhookRequest(BaseModel):
    resource: BitbucketDCResourceIdentifier


class BitbucketDCWebhookEnrollmentResult(BaseModel):
    project_key: str
    repo_slug: str
    success: bool
    error: str | None
    webhook_url: str | None
    webhook_secret: str | None
    webhook_name: str
    events: list[str]


class UpdateBitbucketDCWebhookIdRequest(BaseModel):
    resource: BitbucketDCResourceIdentifier
    webhook_id: str


class BitbucketDCWebhookIdUpdateResult(BaseModel):
    project_key: str
    repo_slug: str
    success: bool
    error: str | None


class BitbucketDCWebhookRequest(BaseModel):
    resource: BitbucketDCResourceIdentifier


class BitbucketDCWebhookInstallationResult(BaseModel):
    project_key: str
    repo_slug: str
    success: bool
    error: str | None
    webhook_id: str | None


def _normalize_dc_resource(
    resource: BitbucketDCResourceIdentifier,
) -> tuple[str, str]:
    project_key = resource.project_key.strip()
    repo_slug = resource.repo_slug.strip()
    if not project_key or not repo_slug:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='project_key and repo_slug are required',
        )
    return project_key, repo_slug


async def _ensure_dc_admin_access(
    bitbucket_dc_service: SaaSBitbucketDCService,
    project_key: str,
    repo_slug: str,
) -> None:
    """Mirror of Cloud's ``_ensure_admin_access``.

    The DC ``user_has_admin_access`` implementation fails open on
    introspection errors (see its docstring), so this check is genuinely
    gating only when the BBDC permissions endpoint returns a clean answer.
    The webhook write itself will surface a 403 if the user really lacks
    admin — that's the true authoritative check.
    """
    if not await bitbucket_dc_service.user_has_admin_access(project_key, repo_slug):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='User does not have admin access to this repository',
        )


async def _get_or_create_dc_webhook(
    bitbucket_dc_service: SaaSBitbucketDCService,
    project_key: str,
    repo_slug: str,
    webhook_secret: str,
) -> str | None:
    """Idempotent webhook (re)installation against BBDC.

    Looks up an existing webhook by URL, updates it with the freshly
    rotated secret if found, or creates a new one. Matches the Cloud
    ``_get_or_create_cloud_webhook`` shape so the route handlers stay
    symmetric.
    """
    (
        webhook_exists,
        webhook_id,
    ) = await bitbucket_dc_service.check_webhook_exists_on_repository(
        project_key, repo_slug, BITBUCKET_DC_WEBHOOK_URL
    )
    if webhook_exists and webhook_id:
        return await bitbucket_dc_service.update_repository_webhook(
            owner=project_key,
            repo_slug=repo_slug,
            webhook_id=webhook_id,
            name=BITBUCKET_DC_WEBHOOK_NAME,
            webhook_url=BITBUCKET_DC_WEBHOOK_URL,
            webhook_secret=webhook_secret,
            events=BITBUCKET_DC_WEBHOOK_EVENTS,
        )

    return await bitbucket_dc_service.create_repository_webhook(
        owner=project_key,
        repo_slug=repo_slug,
        name=BITBUCKET_DC_WEBHOOK_NAME,
        webhook_url=BITBUCKET_DC_WEBHOOK_URL,
        webhook_secret=webhook_secret,
        events=BITBUCKET_DC_WEBHOOK_EVENTS,
    )


def _extract_repo_identity(payload_data: dict) -> tuple[str, str]:
    """Pull ``(project_key, repo_slug)`` out of a DC webhook payload.

    For PR events DC nests the repository under
    ``pullRequest.toRef.repository``; for repo-level events it lives at
    the top level under ``repository``.
    """
    pull_request = payload_data.get('pullRequest') or {}
    repository = (
        (pull_request.get('toRef') or {}).get('repository')
        or payload_data.get('repository')
        or {}
    )
    project = repository.get('project') or {}
    return project.get('key') or '', repository.get('slug') or ''


def _normalize_bitbucket_dc_event_payload(
    payload_data: dict,
    event_key: str | None,
) -> dict:
    """Ensure automation parsing can identify the Bitbucket DC event key.

    Bitbucket DC sends the event key both in ``X-Event-Key`` and, for normal
    deliveries, in the JSON payload. Some tests and proxies only preserve the
    header, so normalize the payload before forwarding it internally.
    """
    if event_key and not payload_data.get('eventKey'):
        payload_data = dict(payload_data)
        payload_data['eventKey'] = event_key
    return payload_data


async def verify_bitbucket_dc_signature(
    *,
    signature_header: str | None,
    body: bytes,
    project_key: str,
    repo_slug: str,
) -> None:
    """Verify ``X-Hub-Signature`` against the per-repository secret.

    Bitbucket Data Center signs each webhook delivery with HMAC-SHA256 of
    the request body using the secret configured on the repository's
    webhook. The header has the form ``sha256=<hex>``. There is no
    per-installation UUID header on DC, so we look up the secret by the
    ``(project_key, repo_slug)`` carried in the payload.
    """
    if not project_key or not repo_slug:
        raise HTTPException(
            status_code=403,
            detail='Missing repository identity in payload',
        )

    if IS_LOCAL_DEPLOYMENT:
        webhook_secret: str | None = 'localdeploymentwebhooktesttoken'
    else:
        webhook_secret = await webhook_store.get_webhook_secret(
            project_key=project_key, repo_slug=repo_slug
        )

    if not webhook_secret:
        raise HTTPException(
            status_code=403,
            detail='No webhook secret found for repository',
        )

    if IS_LOCAL_DEPLOYMENT and signature_header in (
        None,
        'localdeploymentwebhooktesttoken',
    ):
        return

    if not signature_header:
        raise HTTPException(status_code=403, detail='Missing X-Hub-Signature header')

    expected = (
        'sha256=' + hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    )
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=403, detail="Request signatures didn't match!")


@bitbucket_dc_integration_router.get('/bitbucket-dc/resources')
async def get_bitbucket_dc_resources(
    user_id: str = Depends(require_permission(Permission.MANAGE_INTEGRATIONS)),
) -> BitbucketDCResourcesResponse:
    """List Bitbucket DC repositories visible to the user with enrollment status."""
    try:
        bitbucket_dc_service = SaaSBitbucketDCService(external_auth_id=user_id)
        repositories = await bitbucket_dc_service.get_all_repositories(
            sort='updated', app_mode=AppMode.SAAS
        )

        repo_identities: list[tuple[str, str]] = []
        for repo in repositories:
            parts = repo.full_name.split('/', 1)
            if len(parts) != 2:
                logger.warning(
                    f'[Bitbucket DC] Skipping repo with unexpected full_name: {repo.full_name}'
                )
                continue
            repo_identities.append((parts[0], parts[1]))

        webhook_map = await webhook_store.get_webhooks_by_repos(repo_identities)

        resources: list[BitbucketDCResourceWithWebhookStatus] = []
        for project_key, repo_slug in repo_identities:
            webhook = webhook_map.get((project_key, repo_slug))
            resources.append(
                BitbucketDCResourceWithWebhookStatus(
                    project_key=project_key,
                    repo_slug=repo_slug,
                    name=repo_slug,
                    full_name=f'{project_key}/{repo_slug}',
                    webhook_enrolled=bool(webhook and webhook.webhook_secret),
                    webhook_id=webhook.webhook_id if webhook else None,
                    webhook_secret_set=bool(webhook and webhook.webhook_secret),
                    installed_by_user_id=webhook.user_id if webhook else None,
                    last_synced=(
                        webhook.last_synced.isoformat()
                        if webhook and webhook.last_synced
                        else None
                    ),
                )
            )

        return BitbucketDCResourcesResponse(resources=resources)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error retrieving Bitbucket DC resources: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve Bitbucket DC resources',
        )


@bitbucket_dc_integration_router.post('/bitbucket-dc/enroll-webhook')
async def enroll_bitbucket_dc_webhook(
    body: EnrollBitbucketDCWebhookRequest,
    user_id: str = Depends(require_permission(Permission.MANAGE_INTEGRATIONS)),
) -> BitbucketDCWebhookEnrollmentResult:
    """Create or rotate the local enrollment state for a BBDC repo webhook."""
    project_key = body.resource.project_key.strip()
    repo_slug = body.resource.repo_slug.strip()
    if not project_key or not repo_slug:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='project_key and repo_slug are required',
        )

    webhook_secret = secrets.token_urlsafe(32)

    try:
        await webhook_store.upsert_webhook_enrollment(
            project_key=project_key,
            repo_slug=repo_slug,
            user_id=user_id,
            webhook_secret=webhook_secret,
        )

        return BitbucketDCWebhookEnrollmentResult(
            project_key=project_key,
            repo_slug=repo_slug,
            success=True,
            error=None,
            webhook_url=BITBUCKET_DC_WEBHOOK_URL,
            webhook_secret=webhook_secret,
            webhook_name=BITBUCKET_DC_WEBHOOK_NAME,
            events=BITBUCKET_DC_WEBHOOK_EVENTS,
        )

    except Exception as e:
        logger.exception(f'Error enrolling Bitbucket DC webhook: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to enroll Bitbucket DC webhook',
        )


@bitbucket_dc_integration_router.patch('/bitbucket-dc/webhook-id')
async def update_bitbucket_dc_webhook_id(
    body: UpdateBitbucketDCWebhookIdRequest,
    user_id: str = Depends(require_permission(Permission.MANAGE_INTEGRATIONS)),
) -> BitbucketDCWebhookIdUpdateResult:
    """Record the numeric BBDC webhook id after an admin creates it manually."""
    project_key = body.resource.project_key.strip()
    repo_slug = body.resource.repo_slug.strip()
    webhook_id = body.webhook_id.strip()
    if not project_key or not repo_slug or not webhook_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='project_key, repo_slug, and webhook_id are required',
        )

    try:
        updated = await webhook_store.update_webhook_id(
            project_key=project_key,
            repo_slug=repo_slug,
            webhook_id=webhook_id,
        )
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Bitbucket DC webhook enrollment not found',
            )

        logger.info(
            '[Bitbucket DC] Webhook id recorded',
            extra={
                'user_id': user_id,
                'project_key': project_key,
                'repo_slug': repo_slug,
                'webhook_id': webhook_id,
            },
        )
        return BitbucketDCWebhookIdUpdateResult(
            project_key=project_key,
            repo_slug=repo_slug,
            success=True,
            error=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error updating Bitbucket DC webhook id: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update Bitbucket DC webhook id',
        )


@bitbucket_dc_integration_router.post('/bitbucket-dc/reinstall-webhook')
async def reinstall_bitbucket_dc_webhook(
    body: BitbucketDCWebhookRequest,
    user_id: str = Depends(require_permission(Permission.MANAGE_INTEGRATIONS)),
) -> BitbucketDCWebhookInstallationResult:
    """Install or reinstall the webhook for a BBDC repo via the REST API.

    Replaces the manual paste-the-secret flow with a single call: rotates a
    fresh shared secret, idempotently creates or updates the webhook on
    BBDC, and persists the resulting ``webhook_id`` + secret. Requires the
    caller's BBDC OAuth token to have ``REPO_ADMIN`` scope (the Keycloak
    BBDC IDP requests this scope once the chart enables webhook lifecycle).
    """
    project_key, repo_slug = _normalize_dc_resource(body.resource)
    bitbucket_dc_service = SaaSBitbucketDCService(external_auth_id=user_id)

    try:
        await _ensure_dc_admin_access(bitbucket_dc_service, project_key, repo_slug)
        webhook_secret = secrets.token_urlsafe(32)
        webhook_id = await _get_or_create_dc_webhook(
            bitbucket_dc_service, project_key, repo_slug, webhook_secret
        )
        if not webhook_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to install Bitbucket DC webhook',
            )

        await webhook_store.upsert_webhook_enrollment(
            project_key=project_key,
            repo_slug=repo_slug,
            user_id=user_id,
            webhook_id=webhook_id,
            webhook_secret=webhook_secret,
        )

        # Audit log so operators can correlate UI clicks with BBDC
        # webhook state changes when reviewing security incidents — the
        # ``user_has_admin_access`` pre-check is intentionally permissive
        # (BBDC's API is the authoritative auth boundary), so the audit
        # trail lives here.
        logger.info(
            '[Bitbucket DC] Webhook installed',
            extra={
                'user_id': user_id,
                'project_key': project_key,
                'repo_slug': repo_slug,
                'webhook_id': webhook_id,
            },
        )

        return BitbucketDCWebhookInstallationResult(
            project_key=project_key,
            repo_slug=repo_slug,
            success=True,
            error=None,
            webhook_id=webhook_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error installing Bitbucket DC webhook: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to install Bitbucket DC webhook',
        )


@bitbucket_dc_integration_router.post('/bitbucket-dc/uninstall-webhook')
async def uninstall_bitbucket_dc_webhook(
    body: BitbucketDCWebhookRequest,
    user_id: str = Depends(require_permission(Permission.MANAGE_INTEGRATIONS)),
) -> BitbucketDCWebhookInstallationResult:
    """Delete the webhook on BBDC and drop the local enrollment row.

    Looks up the webhook by URL on BBDC (in case the stored ``webhook_id``
    drifted from reality) and calls ``DELETE .../webhooks/{id}``; then
    removes the ``bitbucket_dc_webhook`` row so the resource flips back to
    "not enrolled" in the UI. If no webhook exists on either side this is
    treated as a no-op success — uninstall is idempotent.
    """
    project_key, repo_slug = _normalize_dc_resource(body.resource)
    bitbucket_dc_service = SaaSBitbucketDCService(external_auth_id=user_id)

    try:
        await _ensure_dc_admin_access(bitbucket_dc_service, project_key, repo_slug)
        webhook = await webhook_store.get_webhook_by_repo(project_key, repo_slug)
        (
            provider_exists,
            provider_id,
        ) = await bitbucket_dc_service.check_webhook_exists_on_repository(
            project_key, repo_slug, BITBUCKET_DC_WEBHOOK_URL
        )
        db_id = webhook.webhook_id if webhook else None
        webhook_id = provider_id or db_id
        if provider_id:
            await bitbucket_dc_service.delete_repository_webhook(
                project_key, repo_slug, provider_id
            )
        elif provider_exists:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to locate Bitbucket DC webhook id',
            )

        await webhook_store.delete_webhook_by_repo(
            project_key=project_key,
            repo_slug=repo_slug,
        )

        logger.info(
            '[Bitbucket DC] Webhook uninstalled',
            extra={
                'user_id': user_id,
                'project_key': project_key,
                'repo_slug': repo_slug,
                'webhook_id': webhook_id,
            },
        )

        return BitbucketDCWebhookInstallationResult(
            project_key=project_key,
            repo_slug=repo_slug,
            success=True,
            error=None,
            webhook_id=webhook_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error uninstalling Bitbucket DC webhook: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to uninstall Bitbucket DC webhook',
        )


@bitbucket_dc_integration_router.post('/bitbucket-dc/events')
async def bitbucket_dc_events(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature: str | None = Header(None),
    x_event_key: str | None = Header(None),
    x_request_id: str | None = Header(None),
):
    try:
        body = await request.body()
        payload_data = json.loads(body) if body else {}

        # DC sends a ``diagnostics:ping`` event when the admin clicks the
        # "Test connection" button on the webhook configuration page; it
        # carries no repository payload, so accept it as a no-op.
        if x_event_key == 'diagnostics:ping':
            return JSONResponse(
                status_code=200,
                content={'message': 'Bitbucket DC ping acknowledged.'},
            )

        payload_data = _normalize_bitbucket_dc_event_payload(payload_data, x_event_key)

        project_key, repo_slug = _extract_repo_identity(payload_data)
        await verify_bitbucket_dc_signature(
            signature_header=x_hub_signature,
            body=body,
            project_key=project_key,
            repo_slug=repo_slug,
        )

        pr_id = (payload_data.get('pullRequest') or {}).get('id')
        comment_id = (payload_data.get('comment') or {}).get('id')

        if x_request_id:
            dedup_key = f'bbdc:{x_event_key}:{pr_id}:{comment_id}:{x_request_id}'
        else:
            dedup_hash = hashlib.sha256(body).hexdigest()
            dedup_key = f'bbdc:msg:{dedup_hash}'

        redis = get_redis_client_async()
        created = await redis.set(dedup_key, 1, nx=True, ex=60)
        if not created:
            logger.info('bitbucket_dc_is_duplicate')
            return JSONResponse(
                status_code=200,
                content={'message': 'Duplicate Bitbucket DC event ignored.'},
            )

        installation_id = f'{project_key}/{repo_slug}'
        if AUTOMATION_EVENT_FORWARDING_ENABLED:
            background_tasks.add_task(
                automation_event_service.forward_event,
                provider=ProviderType.BITBUCKET_DATA_CENTER,
                payload=payload_data,
                installation_id=installation_id,
            )

        message = Message(
            source=SourceType.BITBUCKET_DATA_CENTER,
            message={
                'payload': payload_data,
                'event_key': x_event_key,
                'installation_id': installation_id,
            },
        )
        await bitbucket_dc_manager.receive_message(message)

        return JSONResponse(
            status_code=200,
            content={
                'message': 'Bitbucket DC events endpoint reached successfully.',
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error processing Bitbucket DC event: {e}')
        # Surface the exception class name so admins reading DC's webhook
        # delivery UI can correlate with server logs without leaking a full
        # message (which may contain sensitive payload fragments).
        return JSONResponse(
            status_code=400,
            content={'error': 'Invalid payload.', 'error_type': type(e).__name__},
        )
