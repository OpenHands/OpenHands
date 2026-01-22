from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from server.email_validation import get_admin_user_id
from server.routes.org_models import (
    LiteLLMIntegrationError,
    OrgCreate,
    OrgDatabaseError,
    OrgNameExistsError,
    OrgPage,
    OrgResponse,
)
from storage.org_service import OrgService

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
        # Fetch organizations from service layer
        orgs, next_page_id = OrgService.get_user_orgs_paginated(
            user_id=user_id,
            page_id=page_id,
            limit=limit,
        )

        # Convert Org entities to OrgResponse objects
        org_responses = [OrgResponse.from_org(org, credits=None) for org in orgs]

        logger.info(
            'Successfully retrieved organizations',
            extra={
                'user_id': user_id,
                'org_count': len(org_responses),
                'has_more': next_page_id is not None,
            },
        )

        return OrgPage(items=org_responses, next_page_id=next_page_id)

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

        return OrgResponse.from_org(org, credits=credits)
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
