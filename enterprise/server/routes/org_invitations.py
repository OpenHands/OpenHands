"""API routes for organization invitations."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from server.auth.org_context import REJECT_X_ORG_ID_PATH_MISMATCH
from server.routes.org_invitation_models import (
    AcceptInvitationRequest,
    AcceptInvitationResponse,
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
from storage.org_store import OrgStore
from storage.role_store import RoleStore

from openhands.analytics import get_analytics_service
from openhands.app_server.user_auth import get_user_id
from openhands.app_server.utils.logger import openhands_logger as logger

# Router for invitation operations on an organization (requires org_id).
# Every route under this prefix has ``{org_id}`` in its path, so we
# attach REJECT_X_ORG_ID_PATH_MISMATCH at the router level — a request
# with a conflicting ``X-Org-Id`` is rejected before any handler runs.
invitation_router = APIRouter(
    prefix='/api/organizations/{org_id}/members',
    dependencies=[REJECT_X_ORG_ID_PATH_MISMATCH],
)

# Router for accepting invitations (no org_id in path; the target org
# is encoded in the invitation token). X-Org-Id has no meaning here
# and must not influence which invitation is accepted, so the guard
# is intentionally NOT attached.
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

        # Analytics: track team members invited
        try:
            analytics = get_analytics_service()
            if analytics and user_id:
                from storage.user_store import UserStore

                from openhands.analytics.analytics_context import AnalyticsContext

                user_obj = await UserStore.get_user_by_id(user_id)
                ctx = AnalyticsContext(
                    user_id=user_id,
                    consented=user_obj.user_consents_to_analytics is True
                    if user_obj
                    else False,
                    org_id=str(org_id),
                    user=user_obj,
                )
                analytics.track_team_members_invited(
                    ctx=ctx,
                    invited_count=len(invitation_data.emails),
                    successful_count=len(successful),
                    failed_count=len(failed),
                    role=invitation_data.role,
                )
        except Exception:
            logger.exception('analytics:team_members_invited:failed')

        successful_responses = [
            await InvitationResponse.from_invitation(inv) for inv in successful
        ]
        return BatchInvitationResponse(
            successful=successful_responses,
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
async def accept_invitation_redirect(
    token: str,
    request: Request,
):
    """Redirect invitation acceptance to frontend.

    This endpoint is accessed via the link in the invitation email.
    It always redirects to the home page with the invitation token,
    allowing the frontend to handle the acceptance flow via a modal.

    This approach works with SameSite='strict' cookies because:
    - Cross-site navigation (clicking email link) doesn't send cookies
    - But same-origin POST requests (from frontend) DO send cookies

    Args:
        token: The invitation token from the email link
        request: FastAPI request

    Returns:
        RedirectResponse: Redirect to home page with invitation_token query param
    """
    base_url = str(request.base_url).rstrip('/')

    logger.info(
        'Invitation accept: redirecting to frontend for acceptance',
        extra={'token_prefix': token[:10] + '...'},
    )

    return RedirectResponse(f'{base_url}/?invitation_token={token}', status_code=302)


@accept_router.post('/accept', response_model=AcceptInvitationResponse)
async def accept_invitation(
    request_data: AcceptInvitationRequest,
    user_id: str = Depends(get_user_id),
):
    """Accept an organization invitation via authenticated POST request.

    This endpoint is called by the frontend after displaying the acceptance modal.
    Requires authentication - cookies are sent because this is a same-origin request.

    Args:
        request_data: Contains the invitation token
        user_id: Authenticated user ID (from dependency)

    Returns:
        AcceptInvitationResponse: Success response with organization details

    Raises:
        HTTPException 400: Invalid or expired token
        HTTPException 403: Email mismatch
        HTTPException 409: User already a member
    """
    token = request_data.token

    try:
        invitation = await OrgInvitationService.accept_invitation(token, UUID(user_id))

        # Get organization and role details for response
        org = await OrgStore.get_org_by_id(invitation.org_id)
        role = await RoleStore.get_role_by_id(invitation.role_id)

        logger.info(
            'Invitation accepted via API',
            extra={
                'token_prefix': token[:10] + '...',
                'user_id': user_id,
                'org_id': str(invitation.org_id),
            },
        )

        return AcceptInvitationResponse(
            success=True,
            org_id=str(invitation.org_id),
            org_name=org.name if org else '',
            role=role.name if role else '',
        )

    except InvitationExpiredError:
        logger.warning(
            'Invitation accept failed: expired',
            extra={'token_prefix': token[:10] + '...', 'user_id': user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='invitation_expired',
        )

    except InvitationInvalidError as e:
        logger.warning(
            'Invitation accept failed: invalid',
            extra={
                'token_prefix': token[:10] + '...',
                'user_id': user_id,
                'error': str(e),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='invitation_invalid',
        )

    except UserAlreadyMemberError:
        logger.info(
            'Invitation accept: user already member',
            extra={'token_prefix': token[:10] + '...', 'user_id': user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='already_member',
        )

    except EmailMismatchError as e:
        logger.warning(
            'Invitation accept failed: email mismatch',
            extra={
                'token_prefix': token[:10] + '...',
                'user_id': user_id,
                'error': str(e),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='email_mismatch',
        )

    except Exception as e:
        logger.exception(
            'Unexpected error accepting invitation via API',
            extra={
                'token_prefix': token[:10] + '...',
                'user_id': user_id,
                'error': str(e),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred',
        )
