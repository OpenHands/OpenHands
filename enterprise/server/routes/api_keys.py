from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from storage.api_key_store import ApiKeyStore
from storage.lite_llm_manager import LiteLlmManager
from storage.org_member import OrgMember
from storage.org_member_store import OrgMemberStore
from storage.user_store import UserStore

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth import get_user_id


# Helper functions for BYOR API key management
async def get_byor_key_from_db(user_id: str) -> str | None:
    """Get the BYOR key from the database for a user."""
    user = await UserStore.get_user_by_id_async(user_id)
    if not user:
        return None

    current_org_id = user.current_org_id
    current_org_member: OrgMember = None
    for org_member in user.org_members:
        if org_member.org_id == current_org_id:
            current_org_member = org_member
            break
    if not current_org_member:
        return None
    if current_org_member.llm_api_key_for_byor:
        return current_org_member.llm_api_key_for_byor.get_secret_value()
    return None


async def store_byor_key_in_db(user_id: str, key: str) -> None:
    """Store the BYOR key in the database for a user."""
    user = await UserStore.get_user_by_id_async(user_id)
    if not user:
        return None

    current_org_id = user.current_org_id
    current_org_member: OrgMember = None
    for org_member in user.org_members:
        if org_member.org_id == current_org_id:
            current_org_member = org_member
            break
    if not current_org_member:
        return None
    current_org_member.llm_api_key_for_byor = key
    OrgMemberStore.update_org_member(current_org_member)


async def generate_byor_key(user_id: str) -> str | None:
    """Generate a new BYOR key for a user."""

    try:
        user = await UserStore.get_user_by_id_async(user_id)
        if not user:
            return None
        current_org_id = str(user.current_org_id)
        key = await LiteLlmManager.generate_key(
            user_id,
            current_org_id,
            f'BYOR Key - user {user_id}, org {current_org_id}',
            {'type': 'byor'},
        )

        if key:
            logger.info(
                'Successfully generated new BYOR key',
                extra={
                    'user_id': user_id,
                    'key_length': len(key) if key else 0,
                    'key_prefix': key[:10] + '...' if key and len(key) > 10 else key,
                },
            )
            return key
        else:
            logger.error(
                'Failed to generate BYOR LLM API key - no key in response',
                extra={'user_id': user_id},
            )
            return None
    except Exception as e:
        logger.exception(
            'Error generating BYOR key',
            extra={'user_id': user_id, 'error': str(e)},
        )
        return None


async def delete_byor_key_from_litellm(user_id: str, byor_key: str) -> bool:
    """Delete the BYOR key from LiteLLM using the key directly.

    Also attempts to delete by key alias if the key is not found,
    to clean up orphaned aliases that could block key regeneration.
    """
    try:
        # Get user to construct the key alias
        user = await UserStore.get_user_by_id_async(user_id)
        key_alias = None
        if user and user.current_org_id:
            key_alias = f'BYOR Key - user {user_id}, org {user.current_org_id}'

        await LiteLlmManager.delete_key(byor_key, key_alias=key_alias)
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
        api_key = await api_key_store.create_api_key(
            user_id, key_data.name, key_data.expires_at
        )
        # Get the created key details
        keys = await api_key_store.list_api_keys(user_id)
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
        keys = await api_key_store.list_api_keys(user_id)
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
        keys = await api_key_store.list_api_keys(user_id)
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
    """Get the LLM API key for BYOR (Bring Your Own Runtime) for the authenticated user.

    This endpoint validates that the key exists in LiteLLM before returning it.
    If validation fails, it automatically generates a new key to ensure users
    always receive a working key.
    """
    try:
        # Check if the BYOR key exists in the database
        byor_key = await get_byor_key_from_db(user_id)
        if byor_key:
            # Validate that the key is actually registered in LiteLLM
            is_valid = await LiteLlmManager.verify_key(byor_key, user_id)
            if is_valid:
                return {'key': byor_key}
            else:
                # Key exists in DB but is invalid in LiteLLM - regenerate it
                logger.warning(
                    'BYOR key found in database but invalid in LiteLLM - regenerating',
                    extra={
                        'user_id': user_id,
                        'key_prefix': byor_key[:10] + '...'
                        if len(byor_key) > 10
                        else byor_key,
                    },
                )
                # Delete the invalid key from LiteLLM (best effort, don't fail if it doesn't exist)
                await delete_byor_key_from_litellm(user_id, byor_key)
                # Fall through to generate a new key

        # Generate a new key for BYOR (either no key exists or validation failed)
        key = await generate_byor_key(user_id)
        if key:
            # Store the key in the database
            await store_byor_key_in_db(user_id, key)
            logger.info(
                'Successfully generated and stored new BYOR key',
                extra={'user_id': user_id},
            )
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

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
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
