from fastapi import APIRouter, Depends, HTTPException, status
from server.email_validation import get_admin_user_id
from server.routes.org_models import (
    LiteLLMIntegrationError,
    OrgCreate,
    OrgDatabaseError,
    OrgNameExistsError,
    OrgResponse,
)
from storage.org_service import OrgService

from openhands.core.logger import openhands_logger as logger

# Initialize API router
org_router = APIRouter(prefix='/api/organizations')


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

        return OrgResponse(
            id=str(org.id),
            name=org.name,
            contact_name=org.contact_name,
            contact_email=org.contact_email,
            conversation_expiration=org.conversation_expiration,
            agent=org.agent,
            default_max_iterations=org.default_max_iterations,
            security_analyzer=org.security_analyzer,
            confirmation_mode=org.confirmation_mode,
            default_llm_model=org.default_llm_model,
            default_llm_base_url=org.default_llm_base_url,
            remote_runtime_resource_factor=org.remote_runtime_resource_factor,
            enable_default_condenser=org.enable_default_condenser,
            billing_margin=org.billing_margin,
            enable_proactive_conversation_starters=org.enable_proactive_conversation_starters,
            sandbox_base_container_image=org.sandbox_base_container_image,
            sandbox_runtime_container_image=org.sandbox_runtime_container_image,
            org_version=org.org_version,
            mcp_config=org.mcp_config,
            max_budget_per_task=org.max_budget_per_task,
            enable_solvability_analysis=org.enable_solvability_analysis,
            v1_enabled=org.v1_enabled,
            credits=credits,
        )
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
