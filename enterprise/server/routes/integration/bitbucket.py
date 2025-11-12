import os

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from integrations.bitbucket.bitbucket_manager import BitbucketManager
from integrations.models import Message, SourceType
from server.auth.token_manager import TokenManager

from openhands.core.logger import openhands_logger as logger

BITBUCKET_WEBHOOKS_ENABLED = os.environ.get('BITBUCKET_WEBHOOKS_ENABLED', '1') in (
    '1',
    'true',
)

BITBUCKET_WEBHOOK_SECRET = os.environ.get('BITBUCKET_WEBHOOK_SECRET', '')

bitbucket_integration_router = APIRouter(prefix='/integration')
token_manager = TokenManager()
bitbucket_manager = BitbucketManager(token_manager)


def verify_forge_signature(payload: bytes, signature: str):
    """Verify the webhook signature from Forge app."""
    if not signature:
        raise HTTPException(status_code=403, detail='X-Forge-Secret header is missing!')

    # TODO: Upgrade to HMAC-based verification for production
    if signature != BITBUCKET_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail='Invalid webhook secret!')


@bitbucket_integration_router.post('/bitbucket/events')
async def bitbucket_events(
    request: Request,
    x_forge_secret: str = Header(None),
):
    """Handle webhook events from Bitbucket via Forge app."""

    if not BITBUCKET_WEBHOOKS_ENABLED:
        logger.info(
            'Bitbucket webhooks are disabled by BITBUCKET_WEBHOOKS_ENABLED environment variable'
        )
        return JSONResponse(
            status_code=200,
            content={'message': 'Bitbucket webhooks are currently disabled.'},
        )

    try:
        payload = await request.body()

        if not BITBUCKET_WEBHOOK_SECRET:
            logger.error(
                'BITBUCKET_WEBHOOK_SECRET not configured - rejecting request for security'
            )
            raise HTTPException(
                status_code=500, detail='Webhook secret not configured on server'
            )

        verify_forge_signature(payload, x_forge_secret)

        payload_data = await request.json()

        event_data = payload_data.get('event', {})
        context_data = payload_data.get('context', {})

        workspace_slug = event_data.get('workspace', {}).get('slug')
        repo_slug = event_data.get('repository', {}).get('slug')
        pr_id = event_data.get('pullrequest', {}).get('id')

        message_payload = {
            'event': event_data,
            'context': context_data,
        }
        message = Message(source=SourceType.BITBUCKET, message=message_payload)

        await bitbucket_manager.receive_message(message)

        return JSONResponse(
            status_code=200,
            content={
                'message': 'Bitbucket event received and logged successfully.',
                'workspace': workspace_slug,
                'repository': repo_slug,
                'pr_id': pr_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error processing Bitbucket event: {e}')
        return JSONResponse(status_code=400, content={'error': 'Invalid payload.'})
