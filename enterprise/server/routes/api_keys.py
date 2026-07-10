from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, SecretStr, field_validator, model_validator
from server.auth.authorization import get_user_super_role
from server.auth.org_context import EFFECTIVE_ORG_ID
from server.auth.saas_user_auth import SaasUserAuth
from server.constants import BYOR_KEY_ALIAS_PATTERN
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
    org_member.llm_api_key_for_byor = SecretStr(key)
    await OrgMemberStore.update_org_member(org_member)


def _create_byor_key_alias(user_id: str, org_id: str) -> str:
    alias = BYOR_KEY_ALIAS_PATTERN.format(user_id=user_id, org_id=org_id)
    return alias


async def generate_byor_key(user_id: str, org_id: UUID) -> str | None:
    """Generate a new BYOR key for a user in a specific org."""
    try:
        org_id_str = str(org_id)
        key = await LiteLlmManager.generate_key(
            user_id,
            org_id_str,
            _create_byor_key_alias(user_id, org_id_str),
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
    except Exception:
        logger.exception(
            'Error generating BYOR key',
            extra={
                'user_id': user_id,
            },
            stack_info=True,
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
        key_alias = _create_byor_key_alias(user_id, str(org_id))
        await LiteLlmManager.delete_key(byor_key, key_alias=key_alias)
        logger.info(
            'Successfully deleted BYOR key from LiteLLM',
            extra={'user_id': user_id},
        )
        return True
    except Exception:
        logger.exception(
            'Error deleting BYOR key from LiteLLM',
            extra={
                'user_id': user_id,
            },
            stack_info=True,
        )
        return False


# Initialize API router and key store
api_router = APIRouter(prefix='/api/keys')
api_key_store = ApiKeyStore.get_instance()


class ApiKeyCreate(BaseModel):
    name: str | None = None
    not_before: datetime | None = None
    expires_at: datetime | None = None
    # Org the key is bound to. ``None`` (or omitted) creates an *unbound*
    # key whose effective org is resolved per-request via the ``X-Org-Id``
    # header or, as a fallback, the caller's ``user.current_org_id``. When
    # set, the caller must be a member of the requested org.
    org_id: UUID | None = None

    @field_validator('expires_at')
    def validate_expiration(cls, v):
        if v and v < datetime.now(UTC):
            raise ValueError('Expiration date cannot be in the past')
        return v

    @model_validator(mode='after')
    def validate_active_window(self):
        if (
            self.not_before is not None
            and self.expires_at is not None
            and self.not_before >= self.expires_at
        ):
            raise ValueError('not_before must be earlier than expires_at')
        return self


class ApiKeyResponse(BaseModel):
    id: int
    name: str | None = None
    created_at: datetime
    last_used_at: datetime | None = None
    not_before: datetime | None = None
    expires_at: datetime | None = None
    # ``None`` denotes an unbound key (scoped per-request via ``X-Org-Id``).
    org_id: UUID | None = None


class ApiKeyCreateResponse(ApiKeyResponse):
    key: str


class LlmApiKeyResponse(BaseModel):
    key: str | None


class ByorPermittedResponse(BaseModel):
    permitted: bool


class MessageResponse(BaseModel):
    message: str


class CurrentApiKeyResponse(BaseModel):
    """Response model for the current API key endpoint.

    ``org_id`` is the *effective* org id of the current request: the
    key's bound org for org-bound keys, or the resolved org (from the
    ``X-Org-Id`` header or ``user.current_org_id``) for unbound keys.
    ``bound_org_id`` distinguishes the two cases -- it is the org
    persisted on the key, or ``None`` when the key is unbound.
    """

    id: int
    name: str | None
    org_id: str
    bound_org_id: str | None
    user_id: str
    auth_type: str


def api_key_to_response(key: ApiKey) -> ApiKeyResponse:
    """Convert an ApiKey model to an ApiKeyResponse."""
    return ApiKeyResponse(
        id=key.id,
        name=key.name,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        not_before=key.not_before,
        expires_at=key.expires_at,
        org_id=key.org_id,
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
        logger.exception('Error checking BYOR export permission', stack_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to check BYOR export permission',
        ) from e


@api_router.post('', tags=['Keys'])
async def create_api_key(
    key_data: ApiKeyCreate,
    user_id: str = Depends(get_user_id),
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
) -> ApiKeyCreateResponse:
    """Create a new API key for the authenticated user.

    The new key is bound to ``key_data.org_id`` when provided. An *omitted*
    ``org_id`` falls back to the request's effective org (preserving the
    pre-existing API). An *explicit* ``org_id: null`` creates an unbound
    key whose effective org is resolved per-request via the ``X-Org-Id``
    header or, as a fallback, the caller's ``user.current_org_id``. When a
    specific ``org_id`` is supplied, the caller must be a member of that
    org (or hold a super role).
    """
    if 'org_id' in key_data.model_fields_set:
        # Caller expressed an explicit org choice -- ``null`` is meaningful
        # (unbound key), an UUID requires a membership check.
        target_org_id = key_data.org_id
    else:
        # Backwards-compatible default: bind to the effective org.
        target_org_id = effective_org_id

    if target_org_id is not None:
        # Verify the caller is allowed to bind a key to this org.
        try:
            user_uuid = UUID(user_id)
        except (TypeError, ValueError):
            logger.warning('create_api_key_invalid_user_id', extra={'user_id': user_id})
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Invalid user id',
            )

        member = await OrgMemberStore.get_org_member(target_org_id, user_uuid)
        if member is None:
            # Super-role bypass mirrors ``_resolve_org_id``: a user with a
            # cross-org "super" role can bind keys on behalf of orgs they
            # have not joined. The route still requires the explicit
            # ``org_id`` in the request body for this to apply.
            super_role = await get_user_super_role(user_id)
            if super_role is None:
                logger.warning(
                    'create_api_key_not_a_member',
                    extra={'user_id': user_id, 'org_id': str(target_org_id)},
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail='User is not a member of the requested organization',
                )

    try:
        api_key = await api_key_store.create_api_key(
            user_id,
            key_data.name,
            expires_at=key_data.expires_at,
            not_before=key_data.not_before,
            org_id=target_org_id,
            # We've already decided the binding above (explicit null for
            # unbound, the supplied UUID, or the effective org when omitted)
            # so disable the store's current-org fallback.
            use_current_org_fallback=False,
        )
        # Look up the row we just inserted so the response reflects the
        # persisted ``org_id`` (which may be ``None`` for unbound keys).
        # ``list_api_keys`` returns both bound keys for ``target_org_id`` and
        # any unbound keys; matching by name+org disambiguates when the
        # caller reuses a name across different org scopes.
        keys = await api_key_store.list_api_keys(user_id, org_id=target_org_id)
        for key in keys:
            if key.name == key_data.name and key.org_id == target_org_id:
                return ApiKeyCreateResponse(
                    id=key.id,
                    name=key.name,
                    key=api_key,
                    created_at=key.created_at,
                    last_used_at=key.last_used_at,
                    not_before=key.not_before,
                    expires_at=key.expires_at,
                    org_id=key.org_id,
                )
    except HTTPException:
        raise
    except Exception:
        logger.exception('Error creating API key', stack_info=True)
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
        logger.exception('Error listing API keys', stack_info=True)
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
        logger.exception('Error deleting API key', stack_info=True)
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

    Returns the key's bound org (``bound_org_id``, ``None`` for unbound
    keys) and the request's effective org (``org_id``, resolved from the
    ``X-Org-Id`` header or ``user.current_org_id`` for unbound keys).

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

    if saas_user_auth.api_key_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='This endpoint requires API key authentication.',
        )
    # Resolve the effective org so unbound keys report the org they're
    # actually operating in for this request.
    effective_org_id = await saas_user_auth.get_effective_org_id()
    return CurrentApiKeyResponse(
        id=saas_user_auth.api_key_id,
        name=saas_user_auth.api_key_name,
        org_id=str(effective_org_id) if effective_org_id is not None else '',
        bound_org_id=(
            str(saas_user_auth.api_key_org_id)
            if saas_user_auth.api_key_org_id is not None
            else None
        ),
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
        logger.exception('Error retrieving BYOR LLM API key', stack_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve BYOR LLM API key',
        ) from e


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
        logger.exception(
            'HTTP exception during BYOR LLM API key refresh',
            extra={
                'user_id': user_id,
                'status_code': he.status_code,
                'detail': he.detail,
                'exception_type': type(he).__name__,
            },
            stack_info=True,
        )
        raise
    except Exception as e:
        logger.exception(
            'Unexpected error refreshing BYOR LLM API key',
            extra={
                'user_id': user_id,
                'exception_type': type(e).__name__,
            },
            stack_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to refresh BYOR LLM API key',
        ) from e
