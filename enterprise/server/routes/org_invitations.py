"""API routes for organization invitations."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from server.routes.org_invitation_models import (
    BatchInvitationResponse,
    EmailMismatchError,
    InsufficientPermissionError,
    InvitationCreate,
    InvitationExpiredError,
    InvitationFailure,
    InvitationInvalidError,
    InvitationResponse,
    UserAlreadyMemberError,
)
from server.services.org_invitation_service import OrgInvitationService
from server.utils.rate_limit_utils import check_rate_limit_by_user_id

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth import get_user_id
from openhands.server.user_auth.user_auth import get_user_auth

# Router for invitation operations on an organization (requires org_id)
invitation_router = APIRouter(prefix='/api/organizations/{org_id}/members')

# Router for accepting invitations (no org_id required)
accept_router = APIRouter(prefix='/api/organizations/members/invite')


@invitation_router.post(
    '/invite',
    response_model=BatchInvitationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_invitation(
    org_id: UUID,
    invitation_data: InvitationCreate,
    request: Request,
    user_id: str = Depends(get_user_id),
):
    """Create organization invitations for multiple email addresses.

    Sends emails to invitees with secure links to join the organization.
    Supports batch invitations - some may succeed while others fail.

    Permission rules:
    - Only owners and admins can create invitations
    - Admins can only invite with 'member' or 'admin' role (not 'owner')
    - Owners can invite with any role

    Args:
        org_id: Organization UUID
        invitation_data: Invitation details (emails array, role)
        request: FastAPI request
        user_id: Authenticated user ID (from dependency)

    Returns:
        BatchInvitationResponse: Lists of successful and failed invitations

    Raises:
        HTTPException 400: Invalid role or organization not found
        HTTPException 403: User lacks permission to invite
        HTTPException 429: Rate limit exceeded
    """
    # Rate limit: 10 invitations per minute per user (6 seconds between requests)
    await check_rate_limit_by_user_id(
        request=request,
        key_prefix='org_invitation_create',
        user_id=user_id,
        user_rate_limit_seconds=6,
    )

    try:
        successful, failed = await OrgInvitationService.create_invitations_batch(
            org_id=org_id,
            emails=[str(email) for email in invitation_data.emails],
            role_name=invitation_data.role,
            inviter_id=UUID(user_id),
        )

        logger.info(
            'Batch organization invitations created',
            extra={
                'org_id': str(org_id),
                'total_emails': len(invitation_data.emails),
                'successful': len(successful),
                'failed': len(failed),
                'inviter_id': user_id,
            },
        )

        return BatchInvitationResponse(
            successful=[InvitationResponse.from_invitation(inv) for inv in successful],
            failed=[
                InvitationFailure(email=email, error=error) for email, error in failed
            ],
        )

    except InsufficientPermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            'Unexpected error creating batch invitations',
            extra={'org_id': str(org_id), 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )


@accept_router.get('/accept')
async def accept_invitation(
    token: str,
    request: Request,
):
    """Accept an organization invitation via token.

    This endpoint is accessed via the link in the invitation email.

    Flow:
    1. If user is authenticated: Accept invitation directly and redirect to home
    2. If user is not authenticated: Redirect to login page with invitation token
       - Frontend stores token and includes it in OAuth state during login
       - After authentication, keycloak_callback processes the invitation

    Args:
        token: The invitation token from the email link
        request: FastAPI request

    Returns:
        RedirectResponse: Redirect to home page on success, or login page if not authenticated,
                         or home page with error query params on failure
    """
    base_url = str(request.base_url).rstrip('/')

    # Try to get user_id from auth (may not be authenticated)
    user_id = None
    try:
        user_auth = await get_user_auth(request)
        if user_auth:
            user_id = await user_auth.get_user_id()
    except Exception:
        pass

    if not user_id:
        # User not authenticated - redirect to login page with invitation token
        # Frontend will store the token and include it in OAuth state during login
        logger.info(
            'Invitation accept: redirecting unauthenticated user to login',
            extra={'token_prefix': token[:10] + '...'},
        )
        login_url = f'{base_url}/login?invitation_token={token}'
        return RedirectResponse(login_url, status_code=302)

    # User is authenticated - process the invitation directly
    try:
        await OrgInvitationService.accept_invitation(token, UUID(user_id))

        logger.info(
            'Invitation accepted successfully',
            extra={
                'token_prefix': token[:10] + '...',
                'user_id': user_id,
            },
        )

        # Redirect to home page on success
        return RedirectResponse(f'{base_url}/', status_code=302)

    except InvitationExpiredError:
        logger.warning(
            'Invitation accept failed: expired',
            extra={'token_prefix': token[:10] + '...', 'user_id': user_id},
        )
        return RedirectResponse(f'{base_url}/?invitation_expired=true', status_code=302)

    except InvitationInvalidError as e:
        logger.warning(
            'Invitation accept failed: invalid',
            extra={
                'token_prefix': token[:10] + '...',
                'user_id': user_id,
                'error': str(e),
            },
        )
        return RedirectResponse(f'{base_url}/?invitation_invalid=true', status_code=302)

    except UserAlreadyMemberError:
        logger.info(
            'Invitation accept: user already member',
            extra={'token_prefix': token[:10] + '...', 'user_id': user_id},
        )
        return RedirectResponse(f'{base_url}/?already_member=true', status_code=302)

    except EmailMismatchError as e:
        logger.warning(
            'Invitation accept failed: email mismatch',
            extra={
                'token_prefix': token[:10] + '...',
                'user_id': user_id,
                'error': str(e),
            },
        )
        return RedirectResponse(f'{base_url}/?email_mismatch=true', status_code=302)

    except Exception as e:
        logger.exception(
            'Unexpected error accepting invitation',
            extra={
                'token_prefix': token[:10] + '...',
                'user_id': user_id,
                'error': str(e),
            },
        )
        return RedirectResponse(f'{base_url}/?invitation_error=true', status_code=302)
