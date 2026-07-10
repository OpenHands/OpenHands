import csv
import io
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from server.auth.authorization import (
    Permission,
    authorize_permission,
    require_financial_data_access,
    require_permission,
)
from server.auth.org_context import EFFECTIVE_ORG_ID, REJECT_X_ORG_ID_PATH_MISMATCH
from server.routes.org_models import (
    CannotModifySelfError,
    GitOrgAlreadyClaimedError,
    GitOrgClaimRequest,
    GitOrgClaimResponse,
    InsufficientPermissionError,
    InvalidRoleError,
    LastOwnerError,
    LiteLLMIntegrationError,
    MemberUpdateError,
    MeResponse,
    OrgAppSettingsResponse,
    OrgAppSettingsUpdate,
    OrgAuthorizationError,
    OrgBudgetSettingsResponse,
    OrgBudgetSettingsUpdate,
    OrgBudgetThresholdResponse,
    OrgBudgetUserOverrideUpdate,
    OrgBudgetUserResponse,
    OrgConcurrentModificationError,
    OrgConversationPage,
    OrgConversationResponse,
    OrgConversationStats,
    OrgCreate,
    OrgDatabaseError,
    OrgDefaultsSettingsResponse,
    OrgMemberFinancialPage,
    OrgMemberNotFoundError,
    OrgMemberPage,
    OrgMemberResponse,
    OrgMemberUpdate,
    OrgNameExistsError,
    OrgNotFoundError,
    OrgPage,
    OrgResponse,
    OrgUpdate,
    OrgUsageStats,
    OrgUserUsageStats,
    OrphanedUserError,
    RoleNotFoundError,
)
from server.services.org_app_settings_service import (
    OrgAppSettingsService,
    OrgAppSettingsServiceInjector,
)
from server.services.org_budget_service import (
    OrgBudgetService,
    OrgBudgetServiceInjector,
)
from server.services.org_conversation_service import (
    OrgConversationFilterError,
    OrgConversationService,
    OrgConversationServiceInjector,
)
from server.services.org_member_financial_service import OrgMemberFinancialService
from server.services.org_member_service import OrgMemberService
from sqlalchemy.exc import IntegrityError
from storage.org_git_claim_store import OrgGitClaimStore
from storage.org_service import OrgService
from storage.org_store import OrgStore
from storage.user_store import UserStore

from openhands.analytics import get_analytics_service
from openhands.app_server.user_auth import get_user_id
from openhands.app_server.utils.logger import openhands_logger as logger

# Initialize API router
org_router = APIRouter(
    prefix='/api/organizations',
    tags=['Orgs'],
    dependencies=[REJECT_X_ORG_ID_PATH_MISMATCH],
)


_org_budget_service_injector = OrgBudgetServiceInjector()
org_budget_service_dependency = Depends(_org_budget_service_injector.depends)

# Create injector instance and dependency at module level
_org_app_settings_injector = OrgAppSettingsServiceInjector()
org_app_settings_service_dependency = Depends(_org_app_settings_injector.depends)

_org_conversation_service_injector = OrgConversationServiceInjector()
org_conversation_service_dependency = Depends(
    _org_conversation_service_injector.depends
)


@org_router.get('', response_model=OrgPage)
async def list_user_orgs(
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 100,
    user_id: str = Depends(get_user_id),
) -> OrgPage:
    """List organizations for the authenticated user.

    This endpoint returns a paginated list of all organizations that the
    authenticated user is a member of.

    Args:
        page_id: Optional page ID (offset) for pagination
        limit: Maximum number of organizations to return (1-100, default 100)
        user_id: Authenticated user ID (injected by dependency)

    Returns:
        OrgPage: Paginated list of organizations

    Raises:
        HTTPException: 500 if retrieval fails
    """
    logger.info(
        'Listing organizations for user',
        extra={
            'user_id': user_id,
            'page_id': page_id,
            'limit': limit,
        },
    )

    try:
        # Fetch user to get current_org_id
        user = await UserStore.get_user_by_id(user_id)
        current_org_id = (
            str(user.current_org_id) if user and user.current_org_id else None
        )

        # Fetch organizations from service layer
        orgs, next_page_id = await OrgService.get_user_orgs_paginated(
            user_id=user_id,
            page_id=page_id,
            limit=limit,
        )

        # Convert Org entities to OrgResponse objects
        org_responses = [
            OrgResponse.from_org(org, credits=None, user_id=user_id) for org in orgs
        ]

        logger.info(
            'Successfully retrieved organizations',
            extra={
                'user_id': user_id,
                'org_count': len(org_responses),
                'has_more': next_page_id is not None,
            },
        )

        return OrgPage(
            items=org_responses,
            next_page_id=next_page_id,
            current_org_id=current_org_id,
        )

    except Exception as e:
        logger.exception(
            'Unexpected error listing organizations',
            extra={'user_id': user_id, 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve organizations',
        )


@org_router.post('', response_model=OrgResponse, status_code=status.HTTP_201_CREATED)
async def create_org(
    org_data: OrgCreate,
    user_id: str = Depends(require_permission(Permission.CREATE_ORGANIZATION)),
) -> OrgResponse:
    """Create a new organization.

    This endpoint allows authenticated users that hold the
    ``CREATE_ORGANIZATION`` permission to create a new organization. In
    practice this permission is only granted via the ``superadmin``
    role; no regular,
    org-scoped role carries it. The creator is not automatically added
    as a member; a superadmin can provision the initial org users separately.

    Args:
        org_data: Organization creation data
        user_id: Authenticated user ID (injected by ``require_permission``)

    Returns:
        OrgResponse: The created organization details

    Raises:
        HTTPException: 401 if the user is not authenticated
        HTTPException: 403 if the user lacks ``CREATE_ORGANIZATION``
        HTTPException: 409 if organization name already exists
        HTTPException: 500 if creation fails
    """
    logger.info(
        'Creating new organization',
        extra={
            'user_id': user_id,
            'org_name': org_data.name,
        },
    )

    try:
        # Use service layer to create organization
        org = await OrgService.create_org_with_owner(
            name=org_data.name,
            contact_name=org_data.contact_name,
            contact_email=org_data.contact_email,
            user_id=user_id,
            add_creator_as_owner=False,
        )

        # Retrieve credits from LiteLLM
        credits = await OrgService.get_org_credits(user_id, org.id)

        return OrgResponse.from_org(org, credits=credits, user_id=user_id)
    except OrgNameExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except LiteLLMIntegrationError as e:
        logger.error(
            'LiteLLM integration failed',
            extra={'user_id': user_id, 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to create LiteLLM integration',
        )
    except OrgDatabaseError as e:
        logger.error(
            'Database operation failed',
            extra={'user_id': user_id, 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to create organization',
        )
    except Exception as e:
        logger.exception(
            'Unexpected error creating organization',
            extra={'user_id': user_id, 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )


@org_router.get(
    '/{org_id}/settings',
    response_model=OrgDefaultsSettingsResponse,
)
async def get_org_defaults_settings(
    org_id: UUID,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> OrgDefaultsSettingsResponse:
    """Get org-default settings for a specific organization."""
    try:
        org = await OrgService.get_org_by_id(org_id=org_id, user_id=user_id)
        return OrgDefaultsSettingsResponse.from_org(org)
    except OrgNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            'Error getting organization defaults settings',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve organization defaults settings',
        )


@org_router.patch(
    '/{org_id}/settings',
    response_model=OrgDefaultsSettingsResponse,
)
async def update_org_defaults_settings(
    org_id: UUID,
    settings: OrgUpdate,
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
) -> OrgDefaultsSettingsResponse:
    """Update org-default settings for a specific organization."""
    try:
        allowed_fields = {
            'agent_settings_diff',
            'conversation_settings_diff',
            'search_api_key',
            'llm_api_key',
        }
        invalid_fields = settings.updated_fields() - allowed_fields
        if invalid_fields:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    'Only organization default settings fields are supported on '
                    '/api/organizations/{org_id}/settings'
                ),
            )

        updated_org = await OrgService.update_org_with_permissions(
            org_id=org_id,
            update_data=settings,
            user_id=user_id,
        )
        return OrgDefaultsSettingsResponse.from_org(updated_org)
    except OrgNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except OrgDatabaseError as e:
        logger.error(
            'Database error updating organization defaults settings',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update organization defaults settings',
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            'Error updating organization defaults settings',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update organization defaults settings',
        )


@org_router.get(
    '/llm',
    response_model=OrgDefaultsSettingsResponse,
    deprecated=True,
)
async def get_legacy_org_defaults_settings(
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
    user_id: str = Depends(require_permission(Permission.VIEW_LLM_SETTINGS)),
) -> OrgDefaultsSettingsResponse:
    """Get org-default settings through the deprecated ``/llm`` wrapper.

    The org is the request's *effective* org (``X-Org-Id`` > API-key
    binding > ``user.current_org_id``).
    """
    try:
        return await get_org_defaults_settings(org_id=effective_org_id, user_id=user_id)
    except OrgNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            'Error getting legacy organization defaults settings',
            extra={'user_id': user_id, 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve organization defaults settings',
        )


@org_router.post(
    '/llm',
    response_model=OrgDefaultsSettingsResponse,
    deprecated=True,
)
async def update_legacy_org_defaults_settings(
    settings: OrgUpdate,
    effective_org_id: UUID = EFFECTIVE_ORG_ID,
    user_id: str = Depends(require_permission(Permission.EDIT_LLM_SETTINGS)),
) -> OrgDefaultsSettingsResponse:
    """Update org-default settings through the deprecated ``/llm`` wrapper."""
    try:
        if not settings.has_updates():
            org = await OrgStore.get_org_by_id(effective_org_id)
            if not org:
                raise OrgNotFoundError(str(effective_org_id))
            return OrgDefaultsSettingsResponse.from_org(org)
        return await update_org_defaults_settings(
            org_id=effective_org_id,
            settings=settings,
            user_id=user_id,
        )
    except OrgNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            'Error updating legacy organization defaults settings',
            extra={'user_id': user_id, 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update organization defaults settings',
        )


@org_router.get(
    '/app',
    response_model=OrgAppSettingsResponse,
    dependencies=[Depends(require_permission(Permission.MANAGE_APPLICATION_SETTINGS))],
)
async def get_org_app_settings(
    service: OrgAppSettingsService = org_app_settings_service_dependency,
) -> OrgAppSettingsResponse:
    """Get organization app settings for the user's current organization.

    This endpoint retrieves application settings for the authenticated user's
    current organization. Access requires the MANAGE_APPLICATION_SETTINGS permission,
    which is granted to all organization members (member, admin, and owner roles).

    Args:
        service: OrgAppSettingsService (injected by dependency)

    Returns:
        OrgAppSettingsResponse: The organization app settings

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if user lacks MANAGE_APPLICATION_SETTINGS permission
        HTTPException: 404 if current organization not found
    """
    try:
        return await service.get_org_app_settings()
    except OrgNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Current organization not found',
        )
    except Exception as e:
        logger.exception(
            'Unexpected error retrieving organization app settings',
            extra={'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )


@org_router.post(
    '/app',
    response_model=OrgAppSettingsResponse,
)
async def update_org_app_settings(
    update_data: OrgAppSettingsUpdate,
    request: Request,
    service: OrgAppSettingsService = org_app_settings_service_dependency,
    user_id: str = Depends(require_permission(Permission.MANAGE_APPLICATION_SETTINGS)),
) -> OrgAppSettingsResponse:
    """Update organization app settings for the user's current organization.

    Base access requires MANAGE_APPLICATION_SETTINGS (all members). Editing
    org-wide ``registered_marketplaces`` additionally requires EDIT_ORG_SETTINGS
    (admin/owner), since those defaults apply to every member.

    Args:
        update_data: App settings update data
        request: The incoming request (used to resolve the target org for the
            marketplace permission check)
        service: OrgAppSettingsService (injected by dependency)
        user_id: Authenticated user ID (injected by ``require_permission``)

    Returns:
        OrgAppSettingsResponse: The updated organization app settings

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if user lacks the required permission
        HTTPException: 404 if current organization not found
        HTTPException: 409 if a concurrent modification is detected
        HTTPException: 400 if validation errors occur (e.g. duplicate names)
        HTTPException: 500 if update fails
    """
    try:
        # Org-wide marketplaces are admin/owner-only, even though the other
        # settings on this endpoint are member-editable.
        if update_data.registered_marketplaces is not None:
            await authorize_permission(request, user_id, Permission.EDIT_ORG_SETTINGS)
        return await service.update_org_app_settings(update_data)
    except OrgConcurrentModificationError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except OrgNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Current organization not found',
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            'Unexpected error updating organization app settings',
            extra={'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )


@org_router.get(
    '/{org_id}',
    response_model=OrgResponse,
    status_code=status.HTTP_200_OK,
    deprecated=True,
)
async def get_org(
    org_id: UUID,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> OrgResponse:
    """Get organization details by ID through the deprecated detail route."""
    logger.info(
        'Retrieving organization details',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
        },
    )

    try:
        org = await OrgService.get_org_by_id(
            org_id=org_id,
            user_id=user_id,
        )
        credits = await OrgService.get_org_credits(user_id, org.id)
        return OrgResponse.from_org(org, credits=credits, user_id=user_id)
    except OrgNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            'Unexpected error retrieving organization',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )


@org_router.get(
    '/{org_id}/me',
    response_model=MeResponse,
)
async def get_me(
    org_id: UUID,
    user_id: str = Depends(get_user_id),
) -> MeResponse:
    """Get the current user's membership record for an organization.

    Returns the authenticated user's role, status, email, and LLM override
    fields (with masked API keys) within the specified organization.

    Args:
        org_id: Organization ID (UUID)
        user_id: Authenticated user ID (injected by dependency)

    Returns:
        MeResponse: The user's membership data

    Raises:
        HTTPException: 404 if user is not a member or org doesn't exist
        HTTPException: 500 if retrieval fails
    """
    logger.info(
        'Retrieving current member details',
        extra={'user_id': user_id, 'org_id': str(org_id)},
    )

    try:
        user_uuid = UUID(user_id)
        return await OrgMemberService.get_me(org_id, user_uuid)

    except OrgMemberNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Organization with id "{org_id}" not found',
        )
    except RoleNotFoundError as e:
        logger.exception(
            'Role not found for org member',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'role_id': e.role_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )
    except Exception as e:
        logger.exception(
            'Unexpected error retrieving member details',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )


@org_router.delete(
    '/{org_id}',
    status_code=status.HTTP_200_OK,
)
async def delete_org(
    org_id: UUID,
    user_id: str = Depends(require_permission(Permission.DELETE_ORGANIZATION)),
) -> dict:
    """Delete an organization.

    This endpoint permanently deletes an organization and all associated data including
    organization members, conversations, billing data, and external LiteLLM team resources.
    Access requires the DELETE_ORGANIZATION permission, which is granted only to owners.

    Args:
        org_id: Organization ID to delete (UUID)
        user_id: Authenticated user ID (injected by require_permission dependency)

    Returns:
        dict: Confirmation message with deleted organization details

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if user lacks DELETE_ORGANIZATION permission
        HTTPException: 404 if organization not found
        HTTPException: 500 if deletion fails
    """
    logger.info(
        'Organization deletion requested',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
        },
    )

    try:
        # Use service layer to delete organization with cleanup
        deleted_org = await OrgService.delete_org_with_cleanup(
            user_id=user_id,
            org_id=org_id,
        )

        logger.info(
            'Organization deletion completed successfully',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'org_name': deleted_org.name,
            },
        )

        return {
            'message': 'Organization deleted successfully',
            'organization': {
                'id': str(deleted_org.id),
                'name': deleted_org.name,
                'contact_name': deleted_org.contact_name,
                'contact_email': deleted_org.contact_email,
            },
        }

    except OrgNotFoundError as e:
        logger.warning(
            'Organization not found for deletion',
            extra={'user_id': user_id, 'org_id': str(org_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except OrgAuthorizationError as e:
        logger.warning(
            'User not authorized to delete organization',
            extra={'user_id': user_id, 'org_id': str(org_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except OrphanedUserError as e:
        logger.warning(
            'Cannot delete organization: other members would be orphaned',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'orphaned_users': e.user_ids,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except OrgDatabaseError as e:
        logger.error(
            'Database error during organization deletion',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to delete organization',
        )
    except Exception as e:
        logger.exception(
            'Unexpected error during organization deletion',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )


@org_router.patch(
    '/{org_id}',
    response_model=OrgResponse,
)
async def update_org(
    org_id: UUID,
    update_data: OrgUpdate,
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
) -> OrgResponse:
    """Update an existing organization.

    This endpoint updates organization settings. Access requires the EDIT_ORG_SETTINGS
    permission, which is granted to admin and owner roles.

    Args:
        org_id: Organization ID to update (UUID)
        update_data: Organization update data
        user_id: Authenticated user ID (injected by require_permission dependency)

    Returns:
        OrgResponse: The updated organization details

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if user lacks EDIT_ORG_SETTINGS permission
        HTTPException: 404 if organization not found
        HTTPException: 409 if organization name already exists
        HTTPException: 422 if validation errors occur (handled by FastAPI)
        HTTPException: 500 if update fails
    """
    logger.info(
        'Updating organization',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
        },
    )

    try:
        # Use service layer to update organization with permission checks
        updated_org = await OrgService.update_org_with_permissions(
            org_id=org_id,
            update_data=update_data,
            user_id=user_id,
        )

        # Retrieve credits from LiteLLM (following same pattern as create endpoint)
        credits = await OrgService.get_org_credits(user_id, updated_org.id)

        return OrgResponse.from_org(updated_org, credits=credits, user_id=user_id)

    except ValueError as e:
        # Organization not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except OrgNameExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except PermissionError as e:
        # User lacks permission for LLM settings
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except OrgDatabaseError as e:
        logger.error(
            'Database operation failed',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update organization',
        )
    except Exception as e:
        logger.exception(
            'Unexpected error updating organization',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )


@org_router.get(
    '/{org_id}/members',
)
async def get_org_members(
    org_id: UUID,
    page_id: Annotated[
        str | None,
        Query(title='Optional page offset for pagination'),
    ] = None,
    limit: Annotated[
        int,
        Query(
            title='The max number of results in the page',
            gt=0,
            le=100,
        ),
    ] = 10,
    email: Annotated[
        str | None,
        Query(
            title='Filter members by email (case-insensitive partial match)',
            min_length=1,
            max_length=255,
        ),
    ] = None,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> OrgMemberPage:
    """Get all members of an organization with pagination and optional email filter.

    This endpoint retrieves a paginated list of organization members. Access requires
    the VIEW_ORG_SETTINGS permission, which is granted to all organization members
    (member, admin, and owner roles).

    Args:
        org_id: Organization ID (UUID)
        page_id: Optional page offset for pagination
        limit: Maximum number of members to return (1-100, default 10)
        email: Optional email filter (case-insensitive partial match)
        user_id: Authenticated user ID (injected by require_permission dependency)

    Returns:
        OrgMemberPage: Paginated list of organization members with
            current_page and per_page metadata. Use the /count endpoint
            to get the total count separately.

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if user lacks VIEW_ORG_SETTINGS permission
        HTTPException: 400 if org_id or page_id format is invalid
        HTTPException: 500 if retrieval fails
    """
    try:
        success, error_code, data = await OrgMemberService.get_org_members(
            org_id=org_id,
            current_user_id=UUID(user_id),
            page_id=page_id,
            limit=limit,
            email_filter=email,
        )

        if not success:
            error_map: dict[str | None, tuple[int, str]] = {
                'not_a_member': (
                    status.HTTP_403_FORBIDDEN,
                    'You are not a member of this organization',
                ),
                'invalid_page_id': (
                    status.HTTP_400_BAD_REQUEST,
                    'Invalid page_id format',
                ),
                None: (
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    'An error occurred',
                ),
            }
            status_code, detail = error_map.get(
                error_code,
                (status.HTTP_500_INTERNAL_SERVER_ERROR, 'An error occurred'),
            )
            raise HTTPException(status_code=status_code, detail=detail)

        if data is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Failed to retrieve members',
            )

        return data

    except HTTPException:
        raise
    except ValueError:
        logger.exception('Invalid UUID format')
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid organization ID format',
        )
    except Exception:
        logger.exception('Error retrieving organization members')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve members',
        )


@org_router.get(
    '/{org_id}/members/count',
)
async def get_org_members_count(
    org_id: UUID,
    email: Annotated[
        str | None,
        Query(
            title='Filter members by email (case-insensitive partial match)',
            min_length=1,
            max_length=255,
        ),
    ] = None,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> int:
    """Get count of organization members with optional email filter.

    This endpoint returns the total count of organization members matching
    the filter criteria. Access requires the VIEW_ORG_SETTINGS permission,
    which is granted to all organization members (member, admin, and owner roles).

    Args:
        org_id: Organization ID (UUID)
        email: Optional email filter (case-insensitive partial match)
        user_id: Authenticated user ID (injected by require_permission dependency)

    Returns:
        int: Total count of organization members matching the filter

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if user lacks VIEW_ORG_SETTINGS permission or is not a member
        HTTPException: 400 if org_id format is invalid
        HTTPException: 500 if retrieval fails
    """
    try:
        return await OrgMemberService.get_org_members_count(
            org_id=org_id,
            current_user_id=UUID(user_id),
            email_filter=email,
        )
    except OrgMemberNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You are not a member of this organization',
        )
    except ValueError:
        logger.exception('Invalid UUID format')
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid organization ID format',
        )
    except Exception:
        logger.exception('Error retrieving organization member count')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve member count',
        )


@org_router.get(
    '/{org_id}/members/financial',
    response_model=OrgMemberFinancialPage,
)
async def get_org_members_financial(
    org_id: UUID,
    page_id: Annotated[
        str | None,
        Query(
            title='Pagination offset encoded as string',
            description='Offset for pagination (e.g., "0", "10", "20")',
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(
            title='Maximum items per page',
            gt=0,
            le=100,
        ),
    ] = 10,
    email: Annotated[
        str | None,
        Query(
            title='Filter members by email (case-insensitive partial match)',
            min_length=1,
            max_length=255,
        ),
    ] = None,
    user_id: str = Depends(require_financial_data_access),
) -> OrgMemberFinancialPage:
    """Get paginated financial data for organization members.

    Returns financial information (lifetime spend, current budget) for all members
    within the specified organization. Access is restricted to:
    - Organization Admins
    - Organization Owners
    - OpenHands members (users with @openhands.dev emails)

    Args:
        org_id: Organization ID (UUID)
        page_id: Optional pagination offset encoded as string
        limit: Maximum items per page (1-100, default 10)
        email: Optional email filter (case-insensitive partial match)
        user_id: Authenticated user ID (injected by require_financial_data_access)

    Returns:
        OrgMemberFinancialPage: Paginated response with member financial data
            - items: List of members with user_id, email, lifetime_spend,
                     current_budget, and max_budget
            - current_page: Current page number (1-indexed)
            - per_page: Items per page
            - next_page_id: Offset for next page, or None if no more pages

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if user lacks access (not admin/owner and not @openhands.dev)
        HTTPException: 400 if page_id is invalid
        HTTPException: 500 if retrieval fails
    """
    logger.info(
        'Getting financial data for organization members',
        extra={
            'org_id': str(org_id),
            'user_id': user_id,
            'page_id': page_id,
            'limit': limit,
            'email_filter': email,
        },
    )

    try:
        return await OrgMemberFinancialService.get_org_members_financial_data(
            org_id=org_id,
            page_id=page_id,
            limit=limit,
            email_filter=email,
        )
    except ValueError as e:
        logger.warning(
            'Invalid page_id for financial data request',
            extra={'org_id': str(org_id), 'page_id': page_id, 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception:
        logger.exception(
            'Error retrieving organization member financial data',
            extra={'org_id': str(org_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve member financial data',
        )


def _build_budget_response(state: dict) -> OrgBudgetSettingsResponse:
    settings = state['settings']
    thresholds = state['thresholds']
    cycle = state['cycle']
    current_spend = state['current_spend']
    monthly_limit = settings.monthly_limit or 0
    percentage = (current_spend / monthly_limit * 100) if monthly_limit else 0.0

    return OrgBudgetSettingsResponse(
        enabled=settings.enabled,
        monthly_limit=settings.monthly_limit,
        litellm_last_sync_at=settings.litellm_last_sync_at,
        litellm_last_sync_status=settings.litellm_last_sync_status,
        litellm_last_sync_error=settings.litellm_last_sync_error,
        reset_day=settings.reset_day,
        slack_channel=settings.slack_channel,
        slack_team_id=settings.slack_team_id,
        default_user_monthly_limit=settings.default_user_monthly_limit,
        cycle_start_at=cycle.start_at,
        cycle_end_at=cycle.end_at,
        current_spend=current_spend,
        current_spend_percentage=round(percentage, 1),
        thresholds=[
            OrgBudgetThresholdResponse(
                id=threshold.id,
                percentage=threshold.percentage,
                email_enabled=threshold.email_enabled,
                slack_enabled=threshold.slack_enabled,
            )
            for threshold in thresholds
        ],
        users=[OrgBudgetUserResponse(**user) for user in state['users']],
        users_total=state['users_total'],
        users_page=state['users_page'],
        users_per_page=state['users_per_page'],
    )


@org_router.get(
    '/{org_id}/budgets',
    response_model=OrgBudgetSettingsResponse,
)
async def get_org_budget_settings(
    org_id: UUID,
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
    users_page: int = Query(1, ge=1),
    users_per_page: int = Query(50, ge=1, le=1000),
    users_search: str | None = Query(None, max_length=200),
    users_status: str | None = Query(None),
    budget_service: OrgBudgetService = org_budget_service_dependency,
) -> OrgBudgetSettingsResponse:
    logger.info(
        'Getting org budget settings',
        extra={'org_id': str(org_id), 'user_id': user_id},
    )
    state = await budget_service.get_budget_state(
        org_id,
        users_page=users_page,
        users_per_page=users_per_page,
        users_search=users_search,
        users_status=users_status,
    )
    return _build_budget_response(state)


@org_router.patch(
    '/{org_id}/budgets',
    response_model=OrgBudgetSettingsResponse,
)
async def update_org_budget_settings(
    org_id: UUID,
    update: OrgBudgetSettingsUpdate,
    user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
    users_page: int = Query(1, ge=1),
    users_per_page: int = Query(50, ge=1, le=1000),
    users_search: str | None = Query(None, max_length=200),
    users_status: str | None = Query(None),
    budget_service: OrgBudgetService = org_budget_service_dependency,
) -> OrgBudgetSettingsResponse:
    logger.info(
        'Updating org budget settings',
        extra={'org_id': str(org_id), 'user_id': user_id},
    )
    state = await budget_service.update_budget_settings(
        org_id,
        update,
        users_page=users_page,
        users_per_page=users_per_page,
        users_search=users_search,
        users_status=users_status,
    )
    return _build_budget_response(state)


@org_router.put(
    '/{org_id}/budgets/overrides/{user_id}',
    response_model=OrgBudgetUserResponse,
)
async def upsert_org_budget_override(
    org_id: UUID,
    user_id: str,
    update: OrgBudgetUserOverrideUpdate,
    current_user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
    budget_service: OrgBudgetService = org_budget_service_dependency,
) -> OrgBudgetUserResponse:
    logger.info(
        'Updating org budget override',
        extra={
            'org_id': str(org_id),
            'user_id': user_id,
            'actor_id': current_user_id,
        },
    )
    await budget_service.upsert_user_override(
        org_id,
        UUID(user_id),
        monthly_limit=update.monthly_limit,
        is_disabled=update.is_disabled,
    )
    user_row = await budget_service.get_user_budget_row(org_id, UUID(user_id))
    if not user_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='User not found in organization',
        )
    return OrgBudgetUserResponse(**user_row)


@org_router.delete(
    '/{org_id}/budgets/overrides/{user_id}',
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_org_budget_override(
    org_id: UUID,
    user_id: str,
    current_user_id: str = Depends(require_permission(Permission.EDIT_ORG_SETTINGS)),
    budget_service: OrgBudgetService = org_budget_service_dependency,
) -> None:
    logger.info(
        'Deleting org budget override',
        extra={
            'org_id': str(org_id),
            'user_id': user_id,
            'actor_id': current_user_id,
        },
    )
    await budget_service.delete_user_override(org_id, UUID(user_id))
    return None


@org_router.delete(
    '/{org_id}/members/{user_id}',
)
async def remove_org_member(
    org_id: UUID,
    user_id: str,
    current_user_id: str = Depends(get_user_id),
):
    """Remove a member from an organization.

    Only owners and admins can remove members:
    - Owners can remove admins and regular users
    - Admins can only remove regular users

    Users cannot remove themselves. The last owner cannot be removed.
    """
    try:
        success, error = await OrgMemberService.remove_org_member(
            org_id=org_id,
            target_user_id=UUID(user_id),
            current_user_id=UUID(current_user_id),
        )

        if not success:
            error_map: dict[str | None, tuple[int, str]] = {
                'not_a_member': (
                    status.HTTP_403_FORBIDDEN,
                    'You are not a member of this organization',
                ),
                'cannot_remove_self': (
                    status.HTTP_403_FORBIDDEN,
                    'Cannot remove yourself from an organization',
                ),
                'member_not_found': (
                    status.HTTP_404_NOT_FOUND,
                    'Member not found in this organization',
                ),
                'insufficient_permission': (
                    status.HTTP_403_FORBIDDEN,
                    'You do not have permission to remove this member',
                ),
                'cannot_remove_last_owner': (
                    status.HTTP_400_BAD_REQUEST,
                    'Cannot remove the last owner of an organization',
                ),
                'removal_failed': (
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    'Failed to remove member',
                ),
                None: (
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    'An error occurred',
                ),
            }
            status_code, detail = error_map.get(
                error,
                (status.HTTP_500_INTERNAL_SERVER_ERROR, 'An error occurred'),
            )
            raise HTTPException(status_code=status_code, detail=detail)

        return {'message': 'Member removed successfully'}

    except HTTPException:
        raise
    except ValueError:
        logger.exception('Invalid UUID format')
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid organization or user ID format',
        )
    except Exception:
        logger.exception('Error removing organization member')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to remove member',
        )


@org_router.post(
    '/{org_id}/switch',
    response_model=OrgResponse,
    status_code=status.HTTP_200_OK,
)
async def switch_org(
    org_id: UUID,
    user_id: str = Depends(get_user_id),
) -> OrgResponse:
    """Switch to a different organization.

    This endpoint allows authenticated users to switch their current active
    organization. The user must be a member of the target organization.

    Args:
        org_id: Organization ID to switch to (UUID)
        user_id: Authenticated user ID (injected by dependency)

    Returns:
        OrgResponse: The organization details that was switched to

    Raises:
        HTTPException: 422 if org_id is not a valid UUID (handled by FastAPI)
        HTTPException: 403 if user is not a member of the organization
        HTTPException: 404 if organization not found
        HTTPException: 500 if switch fails
    """
    logger.info(
        'Switching organization',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
        },
    )

    try:
        # Use service layer to switch organization with membership validation
        org = await OrgService.switch_org(
            user_id=user_id,
            org_id=org_id,
        )

        # Refresh person profile with new active org on org switch
        analytics = get_analytics_service()
        if analytics:
            try:
                from openhands.analytics import resolve_analytics_context

                ctx = await resolve_analytics_context(user_id)

                analytics.set_person_properties(
                    ctx=ctx,
                    properties={
                        'org_id': str(org_id),
                        'org_name': org.name,
                        'plan_tier': None,  # plan_tier not yet on Org model
                    },
                )
            except Exception:
                logger.exception(
                    'orgs:switch_org:analytics:failed',
                    extra={'user_id': user_id, 'org_id': str(org_id)},
                )

        # Retrieve credits from LiteLLM for the new current org
        credits = await OrgService.get_org_credits(user_id, org.id)

        return OrgResponse.from_org(org, credits=credits, user_id=user_id)

    except OrgNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except OrgAuthorizationError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except OrgDatabaseError as e:
        logger.error(
            'Database operation failed during organization switch',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to switch organization',
        )
    except Exception as e:
        logger.exception(
            'Unexpected error switching organization',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )


@org_router.patch(
    '/{org_id}/members/{user_id}',
    response_model=OrgMemberResponse,
)
async def update_org_member(
    org_id: UUID,
    user_id: str,
    update_data: OrgMemberUpdate,
    current_user_id: str = Depends(get_user_id),
) -> OrgMemberResponse:
    """Update a member's role in an organization.

    Permission rules:
    - Admins can change roles of regular members to Admin or Member
    - Admins cannot modify other Admins or Owners
    - Owners can change roles of Admins and Members to any role (Owner, Admin, Member)
    - Owners cannot modify other Owners

    Members cannot modify their own role. The last owner cannot be demoted.
    """
    try:
        return await OrgMemberService.update_org_member(
            org_id=org_id,
            target_user_id=UUID(user_id),
            current_user_id=UUID(current_user_id),
            update_data=update_data,
        )
    except OrgMemberNotFoundError as e:
        # Distinguish between requester not being a member vs target not found
        if str(current_user_id) in str(e):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='You are not a member of this organization',
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Member not found in this organization',
        )
    except CannotModifySelfError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Cannot modify your own role',
        )
    except RoleNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Role configuration error',
        )
    except InvalidRoleError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid role specified',
        )
    except InsufficientPermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You do not have permission to modify this member',
        )
    except LastOwnerError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Cannot demote the last owner of an organization',
        )
    except MemberUpdateError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update member',
        )
    except ValueError:
        logger.exception('Invalid UUID format')
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid organization or user ID format',
        )
    except Exception:
        logger.exception('Error updating organization member')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update member',
        )


@org_router.get(
    '/{org_id}/git-claims',
    response_model=list[GitOrgClaimResponse],
)
async def get_git_claims(
    org_id: UUID,
    user_id: str = Depends(require_permission(Permission.MANAGE_ORG_CLAIMS)),
) -> list[GitOrgClaimResponse]:
    """Get all Git organization claims for an OpenHands organization.

    Only admin and owner roles can view Git organization claims.

    Args:
        org_id: OpenHands organization UUID
        user_id: Authenticated user ID (injected by permission check)

    Returns:
        List of GitOrgClaimResponse with claim details
    """
    try:
        claims = await OrgGitClaimStore.get_claims_by_org_id(org_id=org_id)
        return [
            GitOrgClaimResponse(
                id=str(claim.id),
                org_id=str(claim.org_id),
                provider=claim.provider,
                git_organization=claim.git_organization,
                claimed_by=str(claim.claimed_by),
                claimed_at=claim.claimed_at.isoformat(),
            )
            for claim in claims
        ]
    except Exception:
        logger.exception('Error fetching Git organization claims')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to fetch Git organization claims',
        )


@org_router.post(
    '/{org_id}/git-claims',
    response_model=GitOrgClaimResponse,
    status_code=status.HTTP_201_CREATED,
)
async def claim_git_organization(
    org_id: UUID,
    request: GitOrgClaimRequest,
    user_id: str = Depends(require_permission(Permission.MANAGE_ORG_CLAIMS)),
) -> GitOrgClaimResponse:
    """Claim a Git organization for an OpenHands organization.

    Only admin and owner roles can claim Git organizations.
    A Git organization can only be claimed by one OpenHands organization at a time.

    Args:
        org_id: OpenHands organization UUID
        request: Claim request with provider and git_organization
        user_id: Authenticated user ID (injected by permission check)

    Returns:
        GitOrgClaimResponse with the created claim details

    Raises:
        HTTPException 409: If the Git organization is already claimed
        HTTPException 403: If user lacks permission
    """
    try:
        # Check if this Git org is already claimed (early feedback for the common case)
        existing_claim = await OrgGitClaimStore.get_claim_by_provider_and_git_org(
            provider=request.provider,
            git_organization=request.git_organization,
        )

        if existing_claim:
            raise GitOrgAlreadyClaimedError(
                provider=request.provider,
                git_organization=request.git_organization,
            )

        # Create the claim — the DB unique constraint handles the race condition
        # where two concurrent requests both pass the check above.
        claim = await OrgGitClaimStore.create_claim(
            org_id=org_id,
            provider=request.provider,
            git_organization=request.git_organization,
            claimed_by=UUID(user_id),
        )

        return GitOrgClaimResponse(
            id=str(claim.id),
            org_id=str(claim.org_id),
            provider=claim.provider,
            git_organization=claim.git_organization,
            claimed_by=str(claim.claimed_by),
            claimed_at=claim.claimed_at.isoformat(),
        )

    except GitOrgAlreadyClaimedError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except IntegrityError as e:
        # Only treat the unique constraint violation as a duplicate claim.
        # Other integrity errors (e.g. FK violations) should surface as 500s.
        if 'uq_provider_git_org' in str(e.orig):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(
                    GitOrgAlreadyClaimedError(
                        provider=request.provider,
                        git_organization=request.git_organization,
                    )
                ),
            )
        logger.exception('Integrity error claiming Git organization')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to claim Git organization',
        )
    except Exception:
        logger.exception('Error claiming Git organization')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to claim Git organization',
        )


@org_router.delete(
    '/{org_id}/git-claims/{claim_id}',
    status_code=status.HTTP_200_OK,
)
async def disconnect_git_organization(
    org_id: UUID,
    claim_id: UUID,
    user_id: str = Depends(require_permission(Permission.MANAGE_ORG_CLAIMS)),
) -> dict:
    """Remove a Git organization claim from an OpenHands organization.

    Only admin and owner roles can disconnect Git organization claims.

    Args:
        org_id: OpenHands organization UUID
        claim_id: Claim UUID to remove
        user_id: Authenticated user ID (injected by permission check)

    Returns:
        dict: Confirmation message on successful deletion

    Raises:
        HTTPException 404: If the claim is not found for this organization
        HTTPException 403: If user lacks permission
    """
    try:
        deleted = await OrgGitClaimStore.delete_claim(
            claim_id=claim_id,
            org_id=org_id,
        )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Git organization claim not found',
            )

        return {'message': 'Git organization claim removed successfully'}

    except HTTPException:
        raise
    except Exception:
        logger.exception('Error disconnecting Git organization')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to disconnect Git organization',
        )


@org_router.get(
    '/{org_id}/conversations',
    response_model=OrgConversationPage,
)
async def list_org_conversations(
    org_id: UUID,
    # Search
    search: Annotated[
        str | None,
        Query(
            title='Search text matching conversation name, creator name, email, or sandbox ID'
        ),
    ] = None,
    # Sorting
    sort_by: Annotated[
        str,
        Query(
            title='Field to sort by',
            description='Options: created_at, updated_at, llm_model, accumulated_cost, title',
        ),
    ] = 'updated_at',
    sort_order: Annotated[
        str,
        Query(
            title='Sort order',
            description='Options: desc (default), asc',
        ),
    ] = 'desc',
    # Filters
    execution_status: Annotated[
        list[str] | None,
        Query(
            title='Filter by execution status',
            description='Comma-separated list: idle, running, paused, finished, error, stuck',
        ),
    ] = None,
    sandbox_status: Annotated[
        str | None,
        Query(
            title='Filter by sandbox status',
            description='Filter by sandbox status: STARTING, RUNNING, PAUSED, ERROR, MISSING. '
            'Note: when filtering by sandbox_status, total_items reflects the unfiltered '
            'count since the filter is applied post-query. This may result in pagination '
            'showing fewer items per page than expected.',
        ),
    ] = None,
    time_window: Annotated[
        str | None,
        Query(
            title='Time window filter',
            description='Options: all, 7d, 30d, 90d',
        ),
    ] = None,
    # Pagination
    page: Annotated[
        int,
        Query(
            title='Page number',
            ge=1,
        ),
    ] = 1,
    per_page: Annotated[
        int,
        Query(
            title='Items per page',
            ge=1,
            le=100,
        ),
    ] = 20,
    include_sub_conversations: Annotated[
        bool,
        Query(
            title='If True, include sub-conversations. If False (default), exclude them.'
        ),
    ] = False,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_CONVERSATIONS)),
    service: OrgConversationService = org_conversation_service_dependency,
) -> OrgConversationPage:
    """List all conversations for an organization.

    This endpoint returns a paginated list of all conversations within an organization,
    including details about each conversation such as the creator, timestamps, and metrics.

    **Access Control**: Requires VIEW_ORG_CONVERSATIONS permission (Admin or Owner role).

    Args:
        org_id: Organization UUID
        search: Search text matching conversation name, creator name, email, or sandbox ID
        sort_by: Field to sort by (created_at, updated_at, llm_model, accumulated_cost, title)
        sort_order: Sort order (desc or asc)
        execution_status: Filter by execution status (idle, running, paused, finished, error, stuck)
        sandbox_status: Filter by sandbox status (STARTING, RUNNING, PAUSED, ERROR, MISSING)
        time_window: Time window filter (all, 7d, 30d, 90d)
        page: Page number (1-indexed)
        per_page: Items per page (1-100)
        include_sub_conversations: If True, include sub-conversations in results
        user_id: Authenticated user ID (injected by require_permission dependency)
        service: OrgConversationService instance

    Returns:
        OrgConversationPage: Paginated list of conversations with total count

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if user lacks VIEW_ORG_CONVERSATIONS permission
        HTTPException: 500 if retrieval fails
    """
    logger.info(
        'Listing organization conversations',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
            'search': search,
            'sort_by': sort_by,
            'sort_order': sort_order,
            'execution_status': execution_status,
            'sandbox_status': sandbox_status,
            'time_window': time_window,
            'page': page,
            'per_page': per_page,
        },
    )

    try:
        # Convert single string to list for service compatibility
        sandbox_status_list = [sandbox_status] if sandbox_status else None
        result = await service.list_org_conversations(
            org_id=org_id,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            execution_status=execution_status,
            sandbox_status=sandbox_status_list,
            time_window=time_window,
            page=page,
            per_page=per_page,
            include_sub_conversations=include_sub_conversations,
        )

        logger.info(
            'Successfully retrieved organization conversations',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'count': len(result.items),
                'total_items': result.total_items,
            },
        )

        return result

    except OrgConversationFilterError as e:
        logger.warning(
            'Invalid organization conversation filter',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'error': e.message,
                'error_code': e.error_code,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except Exception as e:
        logger.exception(
            'Unexpected error listing organization conversations',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve organization conversations',
        )


@org_router.get(
    '/{org_id}/conversations/stats',
    response_model=OrgConversationStats,
)
async def get_org_conversation_stats(
    org_id: UUID,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_CONVERSATIONS)),
    service: OrgConversationService = org_conversation_service_dependency,
) -> OrgConversationStats:
    """Get aggregated statistics for organization conversations.

    Returns counts of active conversations, running runtimes, completed conversations,
    and aggregated cost/token usage for the organization.

    **Access Control**: Requires VIEW_ORG_CONVERSATIONS permission (Admin or Owner role).

    Returns:
        OrgConversationStats: Aggregated statistics for the org
    """
    logger.info(
        'Getting organization conversation stats',
        extra={'user_id': user_id, 'org_id': str(org_id)},
    )

    try:
        stats = await service.get_stats(org_id=org_id)

        logger.info(
            'Successfully retrieved organization conversation stats',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'active_conversations': stats.active_conversations,
                'running_runtimes': stats.running_runtimes,
                'total_cost': stats.total_cost,
            },
        )

        return stats

    except Exception as e:
        logger.exception(
            'Unexpected error getting organization conversation stats',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve organization conversation stats',
        )


@org_router.get(
    '/{org_id}/conversations/usage-stats',
    response_model=OrgUsageStats,
)
async def get_org_conversation_usage_stats(
    org_id: UUID,
    time_window: Annotated[
        str | None,
        Query(
            title='Time window filter',
            description='Options: 7d, 30d, 90d, ytd (overrides days when provided)',
        ),
    ] = None,
    days: int | None = Query(
        default=None, ge=1, description='Number of days to look back'
    ),
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_CONVERSATIONS)),
    service: OrgConversationService = org_conversation_service_dependency,
) -> OrgUsageStats:
    """Get detailed usage statistics for organization dashboard.

    Returns detailed metrics including active users, agent runs, token usage,
    daily breakdown, and team usage for the specified time window.

    **Access Control**: Requires VIEW_ORG_CONVERSATIONS permission (Admin or Owner role).

    Args:
        org_id: The organization ID
        time_window: Time window filter (7d, 30d, 90d, ytd)
        days: Number of days to look back (default 7)

    Returns:
        OrgUsageStats: Detailed usage statistics for the org
    """
    now = datetime.now(timezone.utc)
    resolved_days = days or 7

    if time_window:
        if time_window == 'ytd':
            start_of_year = datetime(now.year, 1, 1, tzinfo=timezone.utc)
            resolved_days = max(1, (now - start_of_year).days + 1)
        elif time_window in {'7d', '30d', '90d'}:
            resolved_days = int(time_window[:-1])
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Invalid time_window. Use 7d, 30d, 90d, or ytd.',
            )

    logger.info(
        'Getting organization conversation usage stats',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
            'days': resolved_days,
            'time_window': time_window,
        },
    )

    try:
        stats = await service.get_usage_stats(org_id=org_id, days=resolved_days)

        logger.info(
            'Successfully retrieved organization conversation usage stats',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'days': resolved_days,
                'time_window': time_window,
                'active_users': stats.active_users,
                'agent_runs': stats.agent_runs,
                'total_tokens': stats.total_tokens,
            },
        )

        return stats

    except Exception as e:
        logger.exception(
            'Unexpected error getting organization conversation usage stats',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve organization conversation usage stats',
        )


@org_router.get(
    '/{org_id}/conversations/user-usage',
    response_model=OrgUserUsageStats,
)
async def get_org_conversation_user_usage_stats(
    org_id: UUID,
    limit: int = Query(
        default=500,
        ge=1,
        le=2000,
        description='Maximum number of user rows to return',
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description='Offset for paginated user rows',
    ),
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_CONVERSATIONS)),
    service: OrgConversationService = org_conversation_service_dependency,
) -> OrgUserUsageStats:
    """Get per-user usage metrics for organization dashboard.

    **Access Control**: Requires VIEW_ORG_CONVERSATIONS permission (Admin or Owner role).

    Args:
        org_id: The organization ID
        limit: Maximum number of user rows to return
        offset: Offset for paginated user rows

    Returns:
        OrgUserUsageStats: Usage statistics aggregated by user
    """
    logger.info(
        'Getting organization user usage stats',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
            'limit': limit,
            'offset': offset,
        },
    )

    try:
        stats = await service.get_user_usage_stats(
            org_id=org_id,
            limit=limit,
            offset=offset,
        )

        logger.info(
            'Successfully retrieved organization user usage stats',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'limit': limit,
                'offset': offset,
                'has_more': stats.has_more,
            },
        )

        return stats

    except Exception as e:
        logger.exception(
            'Unexpected error getting organization user usage stats',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve organization user usage stats',
        )


@org_router.get(
    '/{org_id}/conversations/export',
    response_class=StreamingResponse,
)
async def export_org_conversations_csv(
    org_id: UUID,
    search: Annotated[
        str | None,
        Query(
            title='Search text matching conversation name, creator name, email, or sandbox ID'
        ),
    ] = None,
    sort_by: Annotated[
        str,
        Query(title='Field to sort by'),
    ] = 'updated_at',
    sort_order: Annotated[
        str,
        Query(title='Sort order'),
    ] = 'desc',
    execution_status: Annotated[
        list[str] | None,
        Query(title='Filter by execution status'),
    ] = None,
    sandbox_status: Annotated[
        str | None,
        Query(title='Filter by sandbox status'),
    ] = None,
    time_window: Annotated[
        str | None,
        Query(title='Time window filter (7d, 30d, 90d)'),
    ] = None,
    include_sub_conversations: Annotated[
        bool,
        Query(title='If True, include sub-conversations'),
    ] = False,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_CONVERSATIONS)),
    service: OrgConversationService = org_conversation_service_dependency,
) -> StreamingResponse:
    """Export organization conversations as CSV.

    Returns the filtered conversation dataset as a downloadable CSV file.

    **Access Control**: Requires VIEW_ORG_CONVERSATIONS permission (Admin or Owner role).
    """
    logger.info(
        'Exporting organization conversations as CSV',
        extra={'user_id': user_id, 'org_id': str(org_id)},
    )

    try:
        # Fetch all conversations (no pagination for export)
        result = await service.list_org_conversations(
            org_id=org_id,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            execution_status=execution_status,
            sandbox_status=[sandbox_status] if sandbox_status else None,
            time_window=time_window,
            page=1,
            per_page=10000,  # Export up to 10k records
            include_sub_conversations=include_sub_conversations,
        )

        # Build response headers
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f'conversations_export_{timestamp}.csv'
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
        if result.total_items > 10000:
            headers['X-Export-Truncated'] = 'true'

        async def generate():
            output = io.StringIO()
            writer = csv.writer(output)

            # Write header row
            writer.writerow(
                [
                    'id',
                    'title',
                    'llm_model',
                    'agent_kind',
                    'user_id',
                    'user_email',
                    'created_at',
                    'updated_at',
                    'sandbox_id',
                    'sandbox_status',
                    'runtime_url',
                    'execution_status',
                    'selected_repository',
                    'selected_branch',
                    'trigger',
                    'accumulated_cost',
                    'prompt_tokens',
                    'completion_tokens',
                    'total_tokens',
                    'cache_read_tokens',
                    'cache_write_tokens',
                ]
            )

            # Write data rows
            for item in result.items:
                writer.writerow(
                    [
                        item.id,
                        item.title,
                        item.llm_model,
                        item.agent_kind,
                        item.user_id,
                        item.user_email,
                        item.created_at,
                        item.updated_at,
                        item.sandbox_id,
                        item.sandbox_status,
                        item.runtime_url,
                        item.execution_status,
                        item.selected_repository,
                        item.selected_branch,
                        item.trigger,
                        item.accumulated_cost,
                        item.prompt_tokens,
                        item.completion_tokens,
                        item.total_tokens,
                        item.cache_read_tokens,
                        item.cache_write_tokens,
                    ]
                )

            yield output.getvalue()

        return StreamingResponse(
            generate(),
            media_type='text/csv',
            headers=headers,
        )

    except OrgConversationFilterError as e:
        logger.warning(
            'Invalid organization conversation filter for export',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'error': e.message,
                'error_code': e.error_code,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except Exception as e:
        logger.exception(
            'Unexpected error exporting organization conversations',
            extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to export organization conversations',
        )


@org_router.get(
    '/{org_id}/conversations/{conversation_id}',
    response_model=OrgConversationResponse,
)
async def get_org_conversation(
    org_id: UUID,
    conversation_id: str,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_CONVERSATIONS)),
    service: OrgConversationService = org_conversation_service_dependency,
) -> OrgConversationResponse:
    """Get a specific conversation by ID.

    Returns full conversation details including logs/history reference.

    **Access Control**: Requires VIEW_ORG_CONVERSATIONS permission (Admin or Owner role).

    Raises:
        HTTPException: 404 if conversation not found or not in org
    """
    logger.info(
        'Getting organization conversation',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
            'conversation_id': conversation_id,
        },
    )

    try:
        result = await service.get_org_conversation(
            org_id=org_id,
            conversation_id=conversation_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Conversation not found',
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            'Unexpected error getting organization conversation',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'conversation_id': conversation_id,
                'error': str(e),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve conversation',
        )


@org_router.post(
    '/{org_id}/conversations/{conversation_id}/stop',
)
async def stop_org_conversation(
    org_id: UUID,
    conversation_id: str,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_CONVERSATIONS)),
    service: OrgConversationService = org_conversation_service_dependency,
):
    """Stop a running conversation and its runtime.

    Safely terminates active agent loops and shuts down the underlying
    runtime execution cell.

    **Access Control**: Requires VIEW_ORG_CONVERSATIONS permission (Admin or Owner role).

    Returns:
        Success message with stopped conversation details

    Raises:
        HTTPException: 404 if conversation not found
        HTTPException: 409 if conversation is not running
        HTTPException: 503 if runtime stop fails
    """
    logger.info(
        'Stopping organization conversation',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
            'conversation_id': conversation_id,
        },
    )

    try:
        result = await service.stop_conversation(
            org_id=org_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Conversation not found',
            )

        if result.get('error'):
            error_code = result.get('error_code')
            if error_code == 'sandbox_shared':
                status_code = status.HTTP_409_CONFLICT
            elif error_code in {'sandbox_unavailable', 'sandbox_stop_failed'}:
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            else:
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            raise HTTPException(
                status_code=status_code,
                detail=result['error'],
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            'Unexpected error stopping organization conversation',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'conversation_id': conversation_id,
                'error': str(e),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Failed to stop conversation',
        )
