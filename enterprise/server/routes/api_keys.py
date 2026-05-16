from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, field_validator
from server.auth.org_context import EFFECTIVE_ORG_ID
from server.auth.saas_user_auth import SaasUserAuth
from storage.api_key import ApiKey
from storage.api_key_store import ApiKeyStore
from storage.lite_llm_manager import LiteLlmManager
from storage.org_member import OrgMember
from storage.org_member_store import OrgMemberStore
from storage.org_service import OrgService
from storage.user_store import UserStore

from openhands.app_server.user_auth import get_user_auth, get_user_id
from openhands.app_server.user_auth.user_auth import AuthType
from openhands.app_server.utils.logger import openhands_logger as logger


# Helper functions for BYOR API key management
async def get_byor_key_from_db(user_id: str, org_id: UUID) -> str | None:
    """Get the BYOR key from the database for a user in a specific org."""
    user = await UserStore.get_user_by_id(user_id)
    if not user:
        return None

    org_member: OrgMember | None = None
    for member in user.org_members:
        if member.org_id == org_id:
            org_member = member
            break
    if not org_member:
        return None
    if org_member.llm_api_key_for_byor:
        return org_member.llm_api_key_for_byor.get_secret_value()
    return None


async def store_byor_key_in_db(user_id: str, org_id: UUID, key: str) -> None:
    """Store the BYOR key in the database for a user in a specific org."""
    user = await UserStore.get_user_by_id(user_id)
    if not user:
        return None

    org_member: OrgMember | None = None
    for member in user.org_members:
        if member.org_id == org_id:
            org_member = member
            break
    if not org_member:
        return None
    org_member.llm_api_key_for_byor = key
    await OrgMemberStore.update_org_member(org_member)


async def generate_byor_key(user_id: str, org_id: UUID) -> str | None:
    """Generate a new BYOR key for a user in a specific org."""
    try:
        org_id_str = str(org_id)
        key = await LiteLlmManager.generate_key(
            user_id,
            org_id_str,
            f'BYOR Key - user {user_id}, org {org_id_str}',
            {'type': 'byor'},
        )

        logger.info(
            'Successfully generated new BYOR key',
            extra={
                'user_id': user_id,
                'key_length': len(key),
                'key_prefix': key[:10] + '...' if len(key) > 10 else key,
            },
        )
        return key
    except Exception as e:
        logger.exception(
            'Error generating BYOR key',
            extra={'user_id': user_id, 'error': str(e)},
        )
        return None


async def delete_byor_key_from_litellm(
    user_id: str, org_id: UUID, byor_key: str
) -> bool:
    """Delete the BYOR key from LiteLLM using the key directly.

    Also attempts to delete by key alias if the key is not found,
    to clean up orphaned aliases that could block key regeneration.
    """
    try:
        key_alias = f'BYOR Key - user {user_id}, org {org_id}'
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
    created_at: datetime
    last_used_at: datetime | None = None
    expires_at: datetime | None = None


class ApiKeyCreateResponse(ApiKeyResponse):
    key: str


class LlmApiKeyResponse(BaseModel):
    key: str | None


class ByorPermittedResponse(BaseModel):
    permitted: bool


class MessageResponse(BaseModel):
    message: str


class CurrentApiKeyResponse(BaseModel):
    """Response model for the current API key endpoint."""

    id: int
    name: str | None
    org_id: str
    user_id: str
    auth_type: str


def api_key_to_response(key: ApiKey) -> ApiKeyResponse:
    """Convert an ApiKey model to an ApiKeyResponse."""
    return ApiKeyResponse(
        id=key.id,
        name=key.name,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        expires_at=key.expires_at,
    )


@api_router.get('/llm/byor/permitted', tags=['Keys'])
async def check_byor_permitted(
    user_id: str = Depends(get_user_id),
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
) -> ByorPermittedResponse:
    """Check if BYOR key export is permitted for the request's effective org."""
    try:
        permitted = await OrgService.check_byor_export_enabled(
            user_id, org_id=effective_org_id
        )
        return ByorPermittedResponse(permitted=permitted)
    except Exception as e:
        logger.exception(
            'Error checking BYOR export permission', extra={'error': str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to check BYOR export permission',
        )


@api_router.post('', tags=['Keys'])
async def create_api_key(
    key_data: ApiKeyCreate,
    user_id: str = Depends(get_user_id),
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
) -> ApiKeyCreateResponse:
    """Create a new API key bound to the request's effective org."""
    try:
        api_key = await api_key_store.create_api_key(
            user_id,
            key_data.name,
            key_data.expires_at,
            org_id=effective_org_id,
        )
        # Get the created key details
        keys = await api_key_store.list_api_keys(user_id, org_id=effective_org_id)
        for key in keys:
            if key.name == key_data.name:
                return ApiKeyCreateResponse(
                    id=key.id,
                    name=key.name,
                    key=api_key,
                    created_at=key.created_at,
                    last_used_at=key.last_used_at,
                    expires_at=key.expires_at,
                )
    except Exception:
        logger.exception('Error creating API key')
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail='Failed to create API key',
    )


@api_router.get('', tags=['Keys'])
async def list_api_keys(
    user_id: str = Depends(get_user_id),
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
) -> list[ApiKeyResponse]:
    """List API keys for the authenticated user in the effective org."""
    try:
        keys = await api_key_store.list_api_keys(user_id, org_id=effective_org_id)
        return [api_key_to_response(key) for key in keys]
    except Exception:
        logger.exception('Error listing API keys')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to list API keys',
        )


@api_router.delete('/{key_id}', tags=['Keys'])
async def delete_api_key(
    key_id: int,
    user_id: str = Depends(get_user_id),
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
) -> MessageResponse:
    """Delete an API key, scoped to the effective org."""
    try:
        # First, verify the key belongs to the user in this org.
        keys = await api_key_store.list_api_keys(user_id, org_id=effective_org_id)
        key_to_delete = None

        for key in keys:
            if key.id == key_id:
                key_to_delete = key
                break

        if not key_to_delete:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='API key not found',
            )

        # Delete the key
        success = await api_key_store.delete_api_key_by_id(key_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to delete API key',
            )
        return MessageResponse(message='API key deleted successfully')
    except HTTPException:
        raise
    except Exception:
        logger.exception('Error deleting API key')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to delete API key',
        )


@api_router.get('/current', tags=['Keys'])
async def get_current_api_key(
    request: Request,
    user_id: str = Depends(get_user_id),
) -> CurrentApiKeyResponse:
    """Get information about the currently authenticated API key.

    This endpoint returns metadata about the API key used for the current request,
    including the org_id associated with the key. This is useful for API key
    callers who need to know which organization context their key operates in.

    Returns 400 if not authenticated via API key (e.g., using cookie auth).
    """
    user_auth = await get_user_auth(request)

    # Check if authenticated via API key
    if user_auth.get_auth_type() != AuthType.BEARER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='This endpoint requires API key authentication. Not available for cookie-based auth.',
        )

    # In SaaS context, bearer auth always produces SaasUserAuth
    saas_user_auth = cast(SaasUserAuth, user_auth)

    if saas_user_auth.api_key_org_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='This API key was created before organization support. Please regenerate your API key to use this endpoint.',
        )

    return CurrentApiKeyResponse(
        id=saas_user_auth.api_key_id,
        name=saas_user_auth.api_key_name,
        org_id=str(saas_user_auth.api_key_org_id),
        user_id=user_id,
        auth_type=saas_user_auth.auth_type.value,
    )


@api_router.get('/llm/byor', tags=['Keys'])
async def get_llm_api_key_for_byor(
    user_id: str = Depends(get_user_id),
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
) -> LlmApiKeyResponse:
    """Get the LLM API key for BYOR (Bring Your Own Runtime).

    This endpoint validates that the key exists in LiteLLM before returning it.
    If validation fails, it automatically generates a new key to ensure users
    always receive a working key.

    Returns 402 Payment Required if BYOR export is not enabled for the
    request's effective org.
    """
    try:
        if not await OrgService.check_byor_export_enabled(
            user_id, org_id=effective_org_id
        ):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail='BYOR key export is not enabled. Purchase credits to enable this feature.',
            )

        # Check if the BYOR key exists in the database
        byor_key = await get_byor_key_from_db(user_id, effective_org_id)
        if byor_key:
            # Validate that the key is actually registered in LiteLLM
            is_valid = await LiteLlmManager.verify_key(byor_key, user_id)
            if is_valid:
                return LlmApiKeyResponse(key=byor_key)
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
                await delete_byor_key_from_litellm(user_id, effective_org_id, byor_key)
                # Fall through to generate a new key

        # Generate a new key for BYOR (either no key exists or validation failed)
        key = await generate_byor_key(user_id, effective_org_id)
        if key:
            # Store the key in the database
            await store_byor_key_in_db(user_id, effective_org_id, key)
            logger.info(
                'Successfully generated and stored new BYOR key',
                extra={'user_id': user_id},
            )
            return LlmApiKeyResponse(key=key)
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


@api_router.post('/llm/byor/refresh', tags=['Keys'])
async def refresh_llm_api_key_for_byor(
    user_id: str = Depends(get_user_id),
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
) -> LlmApiKeyResponse:
    """Refresh the LLM API key for BYOR (Bring Your Own Runtime).

    Returns 402 Payment Required if BYOR export is not enabled for the
    request's effective org.
    """
    logger.info('Starting BYOR LLM API key refresh', extra={'user_id': user_id})

    try:
        if not await OrgService.check_byor_export_enabled(
            user_id, org_id=effective_org_id
        ):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail='BYOR key export is not enabled. Purchase credits to enable this feature.',
            )

        # Get the existing BYOR key from the database
        existing_byor_key = await get_byor_key_from_db(user_id, effective_org_id)

        # If we have an existing key, delete it from LiteLLM
        if existing_byor_key:
            delete_success = await delete_byor_key_from_litellm(
                user_id, effective_org_id, existing_byor_key
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
        key = await generate_byor_key(user_id, effective_org_id)
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
        await store_byor_key_in_db(user_id, effective_org_id, key)

        logger.info(
            'BYOR LLM API key refresh completed successfully',
            extra={'user_id': user_id},
        )
        return LlmApiKeyResponse(key=key)
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
