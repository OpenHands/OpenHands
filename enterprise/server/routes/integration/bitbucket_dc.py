from __future__ import annotations

import hashlib
import hmac
import json

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from integrations.bitbucket_data_center.bitbucket_dc_manager import BitbucketDCManager
from integrations.models import Message, SourceType
from integrations.utils import IS_LOCAL_DEPLOYMENT
from server.auth.token_manager import TokenManager
from storage.bitbucket_dc_webhook_store import BitbucketDCWebhookStore
from storage.redis import get_redis_client_async

from openhands.app_server.utils.logger import openhands_logger as logger

bitbucket_dc_integration_router = APIRouter(prefix='/integration')

webhook_store = BitbucketDCWebhookStore()
token_manager = TokenManager()
bitbucket_dc_manager = BitbucketDCManager(token_manager)


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


@bitbucket_dc_integration_router.post('/bitbucket-dc/events')
async def bitbucket_dc_events(
    request: Request,
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

        message = Message(
            source=SourceType.BITBUCKET_DATA_CENTER,
            message={
                'payload': payload_data,
                'event_key': x_event_key,
                'installation_id': f'{project_key}/{repo_slug}',
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
