from __future__ import annotations

import hashlib
import hmac
import json

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from integrations.bitbucket.bitbucket_manager import BitbucketManager
from integrations.models import Message, SourceType
from integrations.utils import IS_LOCAL_DEPLOYMENT
from server.auth.token_manager import TokenManager
from storage.bitbucket_webhook_store import BitbucketWebhookStore
from storage.redis import get_redis_client_async

from openhands.app_server.utils.logger import openhands_logger as logger

bitbucket_integration_router = APIRouter(prefix='/integration')

webhook_store = BitbucketWebhookStore()
token_manager = TokenManager()
bitbucket_manager = BitbucketManager(token_manager)


async def verify_bitbucket_signature(
    *,
    signature_header: str | None,
    body: bytes,
    webhook_uuid: str | None,
) -> None:
    """Verify ``X-Hub-Signature`` against the per-installation secret.

    Bitbucket Cloud sends ``X-Hook-Uuid`` (unique per installed webhook)
    and ``X-Hub-Signature: sha256=<hex>`` (only when the workspace admin
    sets a secret on the webhook). The webhook record is keyed by the
    Bitbucket-issued ``webhook_uuid``; ``BitbucketWebhook.webhook_uuid`` is
    unique, so a single-key lookup is sufficient.
    """
    if not webhook_uuid:
        raise HTTPException(status_code=403, detail='Missing X-Hook-Uuid header')

    if IS_LOCAL_DEPLOYMENT:
        webhook_secret: str | None = 'localdeploymentwebhooktesttoken'
    else:
        webhook_secret = await webhook_store.get_webhook_secret(
            webhook_uuid=webhook_uuid
        )

    if not webhook_secret:
        raise HTTPException(
            status_code=403, detail='No webhook secret found for installation'
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


@bitbucket_integration_router.post('/bitbucket/events')
async def bitbucket_events(
    request: Request,
    x_hub_signature: str | None = Header(None),
    x_event_key: str | None = Header(None),
    x_request_uuid: str | None = Header(None),
    x_hook_uuid: str | None = Header(None),
):
    try:
        body = await request.body()
        await verify_bitbucket_signature(
            signature_header=x_hub_signature,
            body=body,
            webhook_uuid=x_hook_uuid,
        )

        payload_data = json.loads(body) if body else {}
        pr_id = (payload_data.get('pullrequest') or {}).get('id')
        comment_id = (payload_data.get('comment') or {}).get('id')

        # Dedup by (event_key, pr_id, comment_id, request_uuid). Falls back to
        # a hash of the body when the request UUID is missing.
        if x_request_uuid:
            dedup_key = f'bb:{x_event_key}:{pr_id}:{comment_id}:{x_request_uuid}'
        else:
            dedup_hash = hashlib.sha256(body).hexdigest()
            dedup_key = f'bitbucket_msg:{dedup_hash}'

        redis = get_redis_client_async()
        created = await redis.set(dedup_key, 1, nx=True, ex=60)
        if not created:
            logger.info('bitbucket_is_duplicate')
            return JSONResponse(
                status_code=200,
                content={'message': 'Duplicate Bitbucket event ignored.'},
            )

        message = Message(
            source=SourceType.BITBUCKET,
            message={
                'payload': payload_data,
                'event_key': x_event_key,
                'installation_id': x_hook_uuid,
            },
        )
        await bitbucket_manager.receive_message(message)

        return JSONResponse(
            status_code=200,
            content={'message': 'Bitbucket events endpoint reached successfully.'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error processing Bitbucket event: {e}')
        return JSONResponse(status_code=400, content={'error': 'Invalid payload.'})
