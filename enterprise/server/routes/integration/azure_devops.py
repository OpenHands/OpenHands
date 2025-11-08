import hashlib
import json
import os

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from integrations.azure_devops.azure_devops_manager import AzureDevOpsManager
from integrations.azure_devops.data_collector import AzureDevOpsDataCollector
from integrations.models import Message, SourceType
from server.auth.token_manager import TokenManager

from openhands.core.logger import openhands_logger as logger
from openhands.server.shared import sio

# Environment variable to disable Azure DevOps webhooks
AZURE_DEVOPS_WEBHOOKS_ENABLED = os.environ.get(
    'AZURE_DEVOPS_WEBHOOKS_ENABLED', '1'
) in ('1', 'true')

azure_devops_integration_router = APIRouter(prefix='/integration')

# Initialize Azure DevOps manager with TokenManager (matching GitHub/GitLab pattern)
token_manager = TokenManager()
data_collector = AzureDevOpsDataCollector()
azure_devops_manager = AzureDevOpsManager(token_manager, data_collector)


async def verify_api_key(
    authorization: str | None,
) -> str:
    """Verify API key from Authorization header and return user_id.

    Args:
        authorization: Authorization header value (format: "Bearer <api_key>")

    Returns:
        user_id: The user ID associated with the API key

    Raises:
        HTTPException: If authentication fails (401 Unauthorized)
    """
    if not authorization:
        logger.warning('[Azure DevOps Webhook] Missing Authorization header')
        raise HTTPException(
            status_code=401,
            detail='Unauthorized: Missing Authorization header',
        )

    # Extract Bearer token
    if not authorization.startswith('Bearer '):
        logger.warning(
            '[Azure DevOps Webhook] Invalid Authorization header format (missing "Bearer " prefix)'
        )
        raise HTTPException(
            status_code=401,
            detail='Unauthorized: Invalid Authorization header format. Expected "Bearer <api_key>"',
        )

    api_key = authorization[7:]  # Remove "Bearer " prefix

    # Validate API key
    from storage.api_key_store import ApiKeyStore

    api_key_store = ApiKeyStore.get_instance()
    user_id = api_key_store.validate_api_key(api_key)

    if not user_id:
        logger.warning('[Azure DevOps Webhook] Invalid or expired API key')
        raise HTTPException(
            status_code=401,
            detail='Unauthorized: Invalid or expired API key',
        )

    logger.info(
        f'[Azure DevOps Webhook] API key validated successfully for user {user_id}'
    )
    return user_id


@azure_devops_integration_router.post('/azure-devops/events')
async def azure_devops_events(
    request: Request,
    authorization: str | None = Header(None),
):
    """Handle Azure DevOps Service Hook webhook events.

    This endpoint receives Service Hook webhooks from Azure DevOps
    for work item and PR comment events. Authentication is done via
    API key in the Authorization header (Bearer token).

    Args:
        request: The incoming HTTP request
        authorization: Authorization header containing Bearer token (API key)
    """
    logger.info('[Azure DevOps Webhook] Received webhook request')
    logger.debug(f'[Azure DevOps Webhook] Request headers: {dict(request.headers)}')
    logger.debug(
        f'[Azure DevOps Webhook] Request method: {request.method}, URL: {request.url}'
    )

    # Check if Azure DevOps webhooks are enabled
    if not AZURE_DEVOPS_WEBHOOKS_ENABLED:
        logger.info(
            '[Azure DevOps Webhook] Webhooks are disabled by AZURE_DEVOPS_WEBHOOKS_ENABLED environment variable'
        )
        return JSONResponse(
            status_code=200,
            content={'message': 'Azure DevOps webhooks are currently disabled.'},
        )

    try:
        logger.debug('[Azure DevOps Webhook] Parsing request body as JSON')
        payload_data = await request.json()
        logger.debug(
            f'[Azure DevOps Webhook] Payload size: {len(str(payload_data))} bytes'
        )

        # Verify API key from Authorization header
        try:
            user_id = await verify_api_key(authorization)
            logger.info(
                f'[Azure DevOps Webhook] API key verified successfully for user {user_id}'
            )
        except HTTPException as e:
            logger.error(
                f'[Azure DevOps Webhook] Authentication failed: {e.status_code} - {e.detail}'
            )
            raise

        # Validate basic payload structure
        if 'eventType' not in payload_data:
            logger.error('[Azure DevOps Webhook] Missing eventType in payload')
            logger.debug(
                f'[Azure DevOps Webhook] Payload keys: {list(payload_data.keys())}'
            )
            return JSONResponse(
                status_code=400,
                content={'error': 'Missing eventType in payload.'},
            )

        event_type = payload_data.get('eventType')
        subscription_id = payload_data.get('subscriptionId')
        notification_id = payload_data.get('notificationId')

        logger.info(
            f'[Azure DevOps Webhook] Event type: {event_type}, '
            f'Subscription ID: {subscription_id}, '
            f'Notification ID: {notification_id}'
        )

        # Log additional event details if available
        if 'resource' in payload_data:
            resource = payload_data['resource']
            logger.debug(
                f"[Azure DevOps Webhook] Resource type: {resource.get('resourceType', 'unknown')}"
            )
            if 'url' in resource:
                logger.debug(f"[Azure DevOps Webhook] Resource URL: {resource['url']}")

        # Deduplication using Redis (native Azure DevOps payload fields)
        # Azure DevOps includes subscriptionId and notificationId in all webhook payloads
        dedup_key = None
        if subscription_id:
            if notification_id:
                dedup_key = f'azure_devops_{subscription_id}_{notification_id}'
                logger.debug(
                    f'[Azure DevOps Webhook] Dedup key (from IDs): {dedup_key}'
                )
            else:
                # Fallback: hash the entire payload
                dedup_json = json.dumps(payload_data, sort_keys=True)
                dedup_hash = hashlib.sha256(dedup_json.encode()).hexdigest()
                dedup_key = f'azure_devops_msg: {dedup_hash}'
                logger.debug(
                    f'[Azure DevOps Webhook] Dedup key (from hash, no notification_id): {dedup_key[:50]}...'
                )
        else:
            # No subscriptionId, hash entire payload
            dedup_json = json.dumps(payload_data, sort_keys=True)
            dedup_hash = hashlib.sha256(dedup_json.encode()).hexdigest()
            dedup_key = f'azure_devops_msg: {dedup_hash}'
            logger.debug(
                f'[Azure DevOps Webhook] Dedup key (from hash, no subscription_id): {dedup_key[:50]}...'
            )

        # Check Redis for duplicate
        logger.debug('[Azure DevOps Webhook] Checking Redis for duplicate event')
        redis = sio.manager.redis
        created = await redis.set(dedup_key, 1, nx=True, ex=60)
        if not created:
            logger.info(
                f'[Azure DevOps Webhook] Duplicate event ignored: {dedup_key}',
                extra={'event_type': event_type, 'subscription_id': subscription_id},
            )
            return JSONResponse(
                status_code=200,
                content={'message': 'Duplicate Azure DevOps event ignored.'},
            )

        logger.info(
            f'[Azure DevOps Webhook] Processing new webhook event: {event_type}',
            extra={'subscription_id': subscription_id},
        )

        # Create message object
        # Pass payload directly - factory methods expect raw Azure DevOps payload
        logger.debug('[Azure DevOps Webhook] Creating Message object')
        message = Message(
            source=SourceType.AZURE_DEVOPS,
            message=payload_data,
        )
        logger.debug(
            f'[Azure DevOps Webhook] Message object created with source: {message.source}'
        )

        # Process the message
        logger.info('[Azure DevOps Webhook] Sending message to azure_devops_manager')
        try:
            await azure_devops_manager.receive_message(message)
            logger.info('[Azure DevOps Webhook] Message processed successfully')
        except Exception as processing_error:
            # Log processing errors but still return 200 to Azure DevOps
            # This ensures webhooks don't fail even if there are issues with user lookup,
            # organization extraction, or other processing steps
            logger.error(
                f'[Azure DevOps Webhook] Error processing message: {processing_error}',
                exc_info=True,
            )
            logger.warning(
                '[Azure DevOps Webhook] Returning success despite processing error to avoid webhook failure'
            )

        logger.info('[Azure DevOps Webhook] Returning success response')
        return JSONResponse(
            status_code=200,
            content={'message': 'Azure DevOps webhook event received.'},
        )

    except HTTPException as e:
        # Re-raise HTTP exceptions (authentication failures)
        logger.error(
            f'[Azure DevOps Webhook] HTTP exception occurred: {e.status_code} - {e.detail}'
        )
        raise e
    except json.JSONDecodeError as e:
        # Return 200 even for JSON errors to prevent webhook failures
        logger.error(f'[Azure DevOps Webhook] JSON decode error: {e}', exc_info=True)
        logger.warning('[Azure DevOps Webhook] Returning success despite JSON error')
        return JSONResponse(
            status_code=200,
            content={'message': 'Azure DevOps webhook event received (invalid JSON).'},
        )
    except Exception as e:
        # Return 200 for all other errors to prevent webhook failures
        logger.exception(f'[Azure DevOps Webhook] Unexpected error: {e}')
        logger.warning(
            '[Azure DevOps Webhook] Returning success despite unexpected error'
        )
        return JSONResponse(
            status_code=200,
            content={
                'message': 'Azure DevOps webhook event received (processing error).'
            },
        )
