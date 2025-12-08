from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from server.config import get_config
from server.constants import LITE_LLM_API_KEY, LITE_LLM_API_URL
from storage.api_key_store import ApiKeyStore
from storage.database import session_maker
from storage.saas_settings_store import SaasSettingsStore

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth import get_user_id
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.http_session import httpx_verify_option


# Helper functions for BYOR API key management
async def get_byor_key_from_db(user_id: str) -> str | None:
    """Get the BYOR key from the database for a user."""
    config = get_config()
    settings_store = SaasSettingsStore(
        user_id=user_id, session_maker=session_maker, config=config
    )

    user_db_settings = await call_sync_from_async(
        settings_store.get_user_settings_by_keycloak_id, user_id
    )
    if user_db_settings and user_db_settings.llm_api_key_for_byor:
        return user_db_settings.llm_api_key_for_byor
    return None


async def store_byor_key_in_db(user_id: str, key: str) -> None:
    """Store the BYOR key in the database for a user."""
    config = get_config()
    settings_store = SaasSettingsStore(
        user_id=user_id, session_maker=session_maker, config=config
    )

    def _update_user_settings():
        with session_maker() as session:
            user_db_settings = settings_store.get_user_settings_by_keycloak_id(
                user_id, session
            )
            if user_db_settings:
                user_db_settings.llm_api_key_for_byor = key
                session.commit()
                logger.info(
                    'Successfully stored BYOR key in user settings',
                    extra={'user_id': user_id},
                )
            else:
                logger.warning(
                    'User settings not found when trying to store BYOR key',
                    extra={'user_id': user_id},
                )

    await call_sync_from_async(_update_user_settings)


async def generate_byor_key(user_id: str) -> str | None:
    """Generate a new BYOR key for a user."""
    if not (LITE_LLM_API_KEY and LITE_LLM_API_URL):
        logger.warning(
            'LiteLLM API configuration not found', extra={'user_id': user_id}
        )
        return None

    try:
        async with httpx.AsyncClient(
            verify=httpx_verify_option(),
            headers={
                'x-goog-api-key': LITE_LLM_API_KEY,
            },
        ) as client:
            response = await client.post(
                f'{LITE_LLM_API_URL}/key/generate',
                json={
                    'user_id': user_id,
                    'metadata': {'type': 'byor'},
                    'key_alias': f'BYOR Key - user {user_id}',
                },
            )
            response.raise_for_status()
            response_json = response.json()
            key = response_json.get('key')

            if key:
                logger.info(
                    'Successfully generated new BYOR key',
                    extra={
                        'user_id': user_id,
                        'key_length': len(key) if key else 0,
                        'key_prefix': key[:10] + '...'
                        if key and len(key) > 10
                        else key,
                    },
                )
                return key
            else:
                logger.error(
                    'Failed to generate BYOR LLM API key - no key in response',
                    extra={'user_id': user_id, 'response_json': response_json},
                )
                return None
    except Exception as e:
        logger.exception(
            'Error generating BYOR key',
            extra={'user_id': user_id, 'error': str(e)},
        )
        return None


async def delete_byor_key_from_litellm(user_id: str, byor_key: str) -> bool:
    """Delete the BYOR key from LiteLLM using the key directly."""
    if not (LITE_LLM_API_KEY and LITE_LLM_API_URL):
        logger.warning(
            'LiteLLM API configuration not found', extra={'user_id': user_id}
        )
        return False

    try:
        async with httpx.AsyncClient(
            verify=httpx_verify_option(),
            headers={
                'x-goog-api-key': LITE_LLM_API_KEY,
            },
        ) as client:
            # Delete the key directly using the key value
            delete_url = f'{LITE_LLM_API_URL}/key/delete'
            delete_payload = {'keys': [byor_key]}

            delete_response = await client.post(delete_url, json=delete_payload)
            delete_response.raise_for_status()
            logger.info(
                'Successfully deleted BYOR key from LiteLLM',
                extra={'user_id': user_id},
            )
            return True
    except Exception as e:
        logger.exception(
            'Error deleting BYOR key from LiteLLM',
            extra={'user_id': user_id, 'error': str(e)},
        )
        return False


# Initialize API router and key store
api_router = APIRouter(prefix='/api/keys')
api_key_store = ApiKeyStore.get_instance()


class ApiKeyCreate(BaseModel):
    name: str | None = None
    expires_at: datetime | None = None

    @field_validator('expires_at')
    def validate_expiration(cls, v):
        if v and v < datetime.now(UTC):
            raise ValueError('Expiration date cannot be in the past')
        return v


class ApiKeyResponse(BaseModel):
    id: int
    name: str | None = None
    created_at: str
    last_used_at: str | None = None
    expires_at: str | None = None


class ApiKeyCreateResponse(ApiKeyResponse):
    key: str


class LlmApiKeyResponse(BaseModel):
    key: str | None


@api_router.post('', response_model=ApiKeyCreateResponse)
async def create_api_key(key_data: ApiKeyCreate, user_id: str = Depends(get_user_id)):
    """Create a new API key for the authenticated user."""
    try:
        api_key = api_key_store.create_api_key(
            user_id, key_data.name, key_data.expires_at
        )
        # Get the created key details
        keys = api_key_store.list_api_keys(user_id)
        for key in keys:
            if key['name'] == key_data.name:
                return {
                    **key,
                    'key': api_key,
                    'created_at': (
                        key['created_at'].isoformat() if key['created_at'] else None
                    ),
                    'last_used_at': (
                        key['last_used_at'].isoformat() if key['last_used_at'] else None
                    ),
                    'expires_at': (
                        key['expires_at'].isoformat() if key['expires_at'] else None
                    ),
                }
    except Exception:
        logger.exception('Error creating API key')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to create API key',
        )


@api_router.get('', response_model=list[ApiKeyResponse])
async def list_api_keys(user_id: str = Depends(get_user_id)):
    """List all API keys for the authenticated user."""
    try:
        keys = api_key_store.list_api_keys(user_id)
        return [
            {
                **key,
                'created_at': (
                    key['created_at'].isoformat() if key['created_at'] else None
                ),
                'last_used_at': (
                    key['last_used_at'].isoformat() if key['last_used_at'] else None
                ),
                'expires_at': (
                    key['expires_at'].isoformat() if key['expires_at'] else None
                ),
            }
            for key in keys
        ]
    except Exception:
        logger.exception('Error listing API keys')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to list API keys',
        )


@api_router.delete('/{key_id}')
async def delete_api_key(key_id: int, user_id: str = Depends(get_user_id)):
    """Delete an API key."""
    try:
        # First, verify the key belongs to the user
        keys = api_key_store.list_api_keys(user_id)
        key_to_delete = None

        for key in keys:
            if key['id'] == key_id:
                key_to_delete = key
                break

        if not key_to_delete:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='API key not found',
            )

        # Delete the key
        success = api_key_store.delete_api_key_by_id(key_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to delete API key',
            )
        return {'message': 'API key deleted successfully'}
    except HTTPException:
        raise
    except Exception:
        logger.exception('Error deleting API key')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to delete API key',
        )


@api_router.get('/llm/byor', response_model=LlmApiKeyResponse)
async def get_llm_api_key_for_byor(user_id: str = Depends(get_user_id)):
    """Get the LLM API key for BYOR (Bring Your Own Runtime) for the authenticated user."""
    try:
        # Check if the BYOR key exists in the database
        byor_key = await get_byor_key_from_db(user_id)
        if byor_key:
            return {'key': byor_key}

        # If not, generate a new key for BYOR
        key = await generate_byor_key(user_id)
        if key:
            # Store the key in the database
            await store_byor_key_in_db(user_id, key)
            return {'key': key}
        else:
            logger.error(
                'Failed to generate new BYOR LLM API key',
                extra={'user_id': user_id},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to generate new BYOR LLM API key',
            )

    except Exception as e:
        logger.exception('Error retrieving BYOR LLM API key', extra={'error': str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve BYOR LLM API key',
        )


@api_router.post('/llm/byor/refresh', response_model=LlmApiKeyResponse)
async def refresh_llm_api_key_for_byor(user_id: str = Depends(get_user_id)):
    """Refresh the LLM API key for BYOR (Bring Your Own Runtime) for the authenticated user."""
    logger.info('Starting BYOR LLM API key refresh', extra={'user_id': user_id})

    try:
        if not (LITE_LLM_API_KEY and LITE_LLM_API_URL):
            logger.warning(
                'LiteLLM API configuration not found', extra={'user_id': user_id}
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='LiteLLM API configuration not found',
            )

        # Get the existing BYOR key from the database
        existing_byor_key = await get_byor_key_from_db(user_id)

        # If we have an existing key, delete it from LiteLLM
        if existing_byor_key:
            delete_success = await delete_byor_key_from_litellm(
                user_id, existing_byor_key
            )
            if not delete_success:
                logger.warning(
                    'Failed to delete existing BYOR key from LiteLLM, continuing with key generation',
                    extra={'user_id': user_id},
                )
        else:
            logger.info(
                'No existing BYOR key found in database, proceeding with key generation',
                extra={'user_id': user_id},
            )

        # Generate a new key
        key = await generate_byor_key(user_id)
        if not key:
            logger.error(
                'Failed to generate new BYOR LLM API key',
                extra={'user_id': user_id},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to generate new BYOR LLM API key',
            )

        # Store the key in the database
        await store_byor_key_in_db(user_id, key)

        logger.info(
            'BYOR LLM API key refresh completed successfully',
            extra={'user_id': user_id},
        )
        return {'key': key}
    except HTTPException as he:
        logger.error(
            'HTTP exception during BYOR LLM API key refresh',
            extra={
                'user_id': user_id,
                'status_code': he.status_code,
                'detail': he.detail,
                'exception_type': type(he).__name__,
            },
        )
        raise
    except Exception as e:
        logger.exception(
            'Unexpected error refreshing BYOR LLM API key',
            extra={
                'user_id': user_id,
                'error': str(e),
                'exception_type': type(e).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to refresh BYOR LLM API key',
        )
