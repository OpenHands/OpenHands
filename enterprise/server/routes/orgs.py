from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from server.auth.authorization import (
    Permission,
    require_permission,
)
from server.email_validation import get_admin_user_id
from server.routes.org_models import (
    CannotModifySelfError,
    InsufficientPermissionError,
    InvalidRoleError,
    LastOwnerError,
    LiteLLMIntegrationError,
    MemberUpdateError,
    MeResponse,
    OrgAuthorizationError,
    OrgCreate,
    OrgDatabaseError,
    OrgMemberNotFoundError,
    OrgMemberPage,
    OrgMemberResponse,
    OrgMemberUpdate,
    OrgNameExistsError,
    OrgNotFoundError,
    OrgPage,
    OrgResponse,
    OrgUpdate,
    OrphanedUserError,
    RoleNotFoundError,
)
from server.services.org_member_service import OrgMemberService
from storage.org_service import OrgService
from storage.user_store import UserStore

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth import get_user_id

# Initialize API router
org_router = APIRouter(prefix='/api/organizations')


@org_router.get('', response_model=OrgPage)
async def list_user_orgs(
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, lte=100),
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
        user = await UserStore.get_user_by_id_async(user_id)
        current_org_id = (
            str(user.current_org_id) if user and user.current_org_id else None
        )

        # Fetch organizations from service layer
        orgs, next_page_id = OrgService.get_user_orgs_paginated(
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
    user_id: str = Depends(get_admin_user_id),
) -> OrgResponse:
    """Create a new organization.

    This endpoint allows authenticated users with @openhands.dev email to create
    a new organization. The user who creates the organization automatically becomes
    its owner.

    Args:
        org_data: Organization creation data
        user_id: Authenticated user ID (injected by dependency)

    Returns:
        OrgResponse: The created organization details

    Raises:
        HTTPException: 403 if user email domain is not @openhands.dev
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


@org_router.get('/{org_id}', response_model=OrgResponse, status_code=status.HTTP_200_OK)
async def get_org(
    org_id: UUID,
    user_id: str = Depends(require_permission(Permission.VIEW_ORG_SETTINGS)),
) -> OrgResponse:
    """Get organization details by ID.

    This endpoint retrieves details for a specific organization. Access requires
    the VIEW_ORG_SETTINGS permission, which is granted to all organization members
    (member, admin, and owner roles).

    Args:
        org_id: Organization ID (UUID)
        user_id: Authenticated user ID (injected by require_permission dependency)

    Returns:
        OrgResponse: The organization details

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if user lacks VIEW_ORG_SETTINGS permission
        HTTPException: 404 if organization not found
        HTTPException: 422 if org_id is not a valid UUID (handled by FastAPI)
        HTTPException: 500 if retrieval fails
    """
    logger.info(
        'Retrieving organization details',
        extra={
            'user_id': user_id,
            'org_id': str(org_id),
        },
    )

    try:
        # Use service layer to get organization with membership validation
        org = await OrgService.get_org_by_id(
            org_id=org_id,
            user_id=user_id,
        )

        # Retrieve credits from LiteLLM
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


@org_router.get('/{org_id}/me', response_model=MeResponse)
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
        return OrgMemberService.get_me(org_id, user_uuid)

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


@org_router.delete('/{org_id}', status_code=status.HTTP_200_OK)
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
            'Cannot delete organization: users would be orphaned',
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


@org_router.patch('/{org_id}', response_model=OrgResponse)
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


@org_router.get('/{org_id}/members')
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
            lte=100,
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
            error_map = {
                'not_a_member': (
                    status.HTTP_403_FORBIDDEN,
                    'You are not a member of this organization',
                ),
                'invalid_page_id': (
                    status.HTTP_400_BAD_REQUEST,
                    'Invalid page_id format',
                ),
            }
            status_code, detail = error_map.get(
                error_code, (status.HTTP_500_INTERNAL_SERVER_ERROR, 'An error occurred')
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


@org_router.get('/{org_id}/members/count')
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


@org_router.delete('/{org_id}/members/{user_id}')
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
            error_map = {
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
            }
            status_code, detail = error_map.get(
                error, (status.HTTP_500_INTERNAL_SERVER_ERROR, 'An error occurred')
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
    '/{org_id}/switch', response_model=OrgResponse, status_code=status.HTTP_200_OK
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


@org_router.patch('/{org_id}/members/{user_id}', response_model=OrgMemberResponse)
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
