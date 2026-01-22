"""
Service class for managing organization operations.
Separates business logic from route handlers.
"""

from uuid import UUID, uuid4
from uuid import UUID as parse_uuid

from server.constants import ORG_SETTINGS_VERSION, get_default_litellm_model
from server.routes.org_models import (
    LiteLLMIntegrationError,
    OrgDatabaseError,
    OrgNameExistsError,
)
from storage.lite_llm_manager import LiteLlmManager
from storage.org import Org
from storage.org_member import OrgMember
from storage.org_member_store import OrgMemberStore
from storage.org_store import OrgStore
from storage.role_store import RoleStore
from storage.user_store import UserStore

from openhands.core.logger import openhands_logger as logger


class OrgService:
    """Service for handling organization-related operations."""

    @staticmethod
    def validate_name_uniqueness(name: str) -> None:
        """
        Validate that organization name is unique.

        Args:
            name: Organization name to validate

        Raises:
            OrgNameExistsError: If organization name already exists
        """
        existing_org = OrgStore.get_org_by_name(name)
        if existing_org is not None:
            raise OrgNameExistsError(name)

    @staticmethod
    async def create_litellm_integration(org_id: UUID, user_id: str) -> dict:
        """
        Create LiteLLM team integration for the organization.

        Args:
            org_id: Organization ID
            user_id: User ID who will own the organization

        Returns:
            dict: LiteLLM settings object

        Raises:
            LiteLLMIntegrationError: If LiteLLM integration fails
        """
        try:
            settings = await UserStore.create_default_settings(
                org_id=str(org_id), user_id=user_id, create_user=False
            )

            if not settings:
                logger.error(
                    'Failed to create LiteLLM settings',
                    extra={'org_id': str(org_id), 'user_id': user_id},
                )
                raise LiteLLMIntegrationError('Failed to create LiteLLM settings')

            logger.debug(
                'LiteLLM integration created',
                extra={'org_id': str(org_id), 'user_id': user_id},
            )
            return settings

        except LiteLLMIntegrationError:
            raise
        except Exception as e:
            logger.exception(
                'Error creating LiteLLM integration',
                extra={'org_id': str(org_id), 'user_id': user_id, 'error': str(e)},
            )
            raise LiteLLMIntegrationError(f'LiteLLM integration failed: {str(e)}')

    @staticmethod
    def create_org_entity(
        org_id: UUID,
        name: str,
        contact_name: str,
        contact_email: str,
    ) -> Org:
        """
        Create an organization entity with basic information.

        Args:
            org_id: Organization UUID
            name: Organization name
            contact_name: Contact person name
            contact_email: Contact email address

        Returns:
            Org: New organization entity (not yet persisted)
        """
        return Org(
            id=org_id,
            name=name,
            contact_name=contact_name,
            contact_email=contact_email,
            org_version=ORG_SETTINGS_VERSION,
            default_llm_model=get_default_litellm_model(),
        )

    @staticmethod
    def apply_litellm_settings_to_org(org: Org, settings: dict) -> None:
        """
        Apply LiteLLM settings to organization entity.

        Args:
            org: Organization entity to update
            settings: LiteLLM settings object
        """
        org_kwargs = OrgStore.get_kwargs_from_settings(settings)
        for key, value in org_kwargs.items():
            if hasattr(org, key):
                setattr(org, key, value)

    @staticmethod
    def get_owner_role():
        """
        Get the owner role from the database.

        Returns:
            Role: The owner role object

        Raises:
            Exception: If owner role not found
        """
        owner_role = RoleStore.get_role_by_name('owner')
        if not owner_role:
            raise Exception('Owner role not found in database')
        return owner_role

    @staticmethod
    def create_org_member_entity(
        org_id: UUID,
        user_id: str,
        role_id: int,
        settings: dict,
    ) -> OrgMember:
        """
        Create an organization member entity.

        Args:
            org_id: Organization UUID
            user_id: User ID (string that will be converted to UUID)
            role_id: Role ID
            settings: LiteLLM settings object

        Returns:
            OrgMember: New organization member entity (not yet persisted)
        """
        org_member_kwargs = OrgMemberStore.get_kwargs_from_settings(settings)
        return OrgMember(
            org_id=org_id,
            user_id=parse_uuid(user_id),
            role_id=role_id,
            status='active',
            **org_member_kwargs,
        )

    @staticmethod
    async def create_org_with_owner(
        name: str,
        contact_name: str,
        contact_email: str,
        user_id: str,
    ) -> Org:
        """
        Create a new organization with the specified user as owner.

        This method orchestrates the complete organization creation workflow:
        1. Validates that the organization name doesn't already exist
        2. Generates a unique organization ID
        3. Creates LiteLLM team integration
        4. Creates the organization entity
        5. Applies LiteLLM settings
        6. Creates owner membership
        7. Persists everything in a transaction

        If database persistence fails, LiteLLM resources are cleaned up (compensation).

        Args:
            name: Organization name (must be unique)
            contact_name: Contact person name
            contact_email: Contact email address
            user_id: ID of the user who will be the owner

        Returns:
            Org: The created organization object

        Raises:
            OrgNameExistsError: If organization name already exists
            LiteLLMIntegrationError: If LiteLLM integration fails
            OrgDatabaseError: If database operations fail
        """
        logger.info(
            'Starting organization creation',
            extra={'user_id': user_id, 'org_name': name},
        )

        # Step 1: Validate name uniqueness (fails early, no cleanup needed)
        OrgService.validate_name_uniqueness(name)

        # Step 2: Generate organization ID
        org_id = uuid4()

        # Step 3: Create LiteLLM integration (external state created)
        settings = await OrgService.create_litellm_integration(org_id, user_id)

        # Steps 4-7: Create entities and persist with compensation
        # If any of these fail, we need to clean up LiteLLM resources
        try:
            # Step 4: Create organization entity
            org = OrgService.create_org_entity(
                org_id=org_id,
                name=name,
                contact_name=contact_name,
                contact_email=contact_email,
            )

            # Step 5: Apply LiteLLM settings
            OrgService.apply_litellm_settings_to_org(org, settings)

            # Step 6: Get owner role and create member entity
            owner_role = OrgService.get_owner_role()
            org_member = OrgService.create_org_member_entity(
                org_id=org_id,
                user_id=user_id,
                role_id=owner_role.id,
                settings=settings,
            )

            # Step 7: Persist in transaction (critical section)
            persisted_org = await OrgService._persist_with_compensation(
                org, org_member, org_id, user_id
            )

            logger.info(
                'Successfully created organization',
                extra={
                    'org_id': str(persisted_org.id),
                    'org_name': persisted_org.name,
                    'user_id': user_id,
                    'role': 'owner',
                },
            )

            return persisted_org

        except OrgDatabaseError:
            # Already handled by _persist_with_compensation, just re-raise
            raise
        except Exception as e:
            # Unexpected error in steps 4-6, need to clean up LiteLLM
            logger.error(
                'Unexpected error during organization creation, initiating cleanup',
                extra={
                    'org_id': str(org_id),
                    'user_id': user_id,
                    'error': str(e),
                },
            )
            await OrgService._handle_failure_with_cleanup(
                org_id, user_id, e, 'Failed to create organization'
            )

    @staticmethod
    async def _persist_with_compensation(
        org: Org,
        org_member: OrgMember,
        org_id: UUID,
        user_id: str,
    ) -> Org:
        """
        Persist organization with compensation on failure.

        If database persistence fails, cleans up LiteLLM resources.

        Args:
            org: Organization entity to persist
            org_member: Organization member entity to persist
            org_id: Organization ID (for cleanup)
            user_id: User ID (for cleanup)

        Returns:
            Org: The persisted organization object

        Raises:
            OrgDatabaseError: If database operations fail
        """
        try:
            persisted_org = OrgStore.persist_org_with_owner(org, org_member)
            return persisted_org

        except Exception as e:
            logger.error(
                'Database persistence failed, initiating LiteLLM cleanup',
                extra={
                    'org_id': str(org_id),
                    'user_id': user_id,
                    'error': str(e),
                },
            )
            await OrgService._handle_failure_with_cleanup(
                org_id, user_id, e, 'Failed to create organization'
            )

    @staticmethod
    async def _handle_failure_with_cleanup(
        org_id: UUID,
        user_id: str,
        original_error: Exception,
        error_message: str,
    ) -> None:
        """
        Handle failure by cleaning up LiteLLM resources and raising appropriate error.

        This method performs compensating transaction and raises OrgDatabaseError.

        Args:
            org_id: Organization ID
            user_id: User ID
            original_error: The original exception that caused the failure
            error_message: Base error message for the exception

        Raises:
            OrgDatabaseError: Always raises with details about the failure
        """
        cleanup_error = await OrgService._cleanup_litellm_resources(org_id, user_id)

        if cleanup_error:
            logger.error(
                'Both operation and cleanup failed',
                extra={
                    'org_id': str(org_id),
                    'user_id': user_id,
                    'original_error': str(original_error),
                    'cleanup_error': str(cleanup_error),
                },
            )
            raise OrgDatabaseError(
                f'{error_message}: {str(original_error)}. '
                f'Cleanup also failed: {str(cleanup_error)}'
            )

        raise OrgDatabaseError(f'{error_message}: {str(original_error)}')

    @staticmethod
    async def _cleanup_litellm_resources(
        org_id: UUID, user_id: str
    ) -> Exception | None:
        """
        Compensating transaction: Clean up LiteLLM resources.

        Deletes the team which should cascade to remove keys and memberships.
        This is a best-effort operation - errors are logged but not raised.

        Args:
            org_id: Organization ID
            user_id: User ID

        Returns:
            Exception | None: Exception if cleanup failed, None if successful
        """
        try:
            await LiteLlmManager.delete_team(str(org_id))

            logger.info(
                'Successfully cleaned up LiteLLM team',
                extra={'org_id': str(org_id), 'user_id': user_id},
            )
            return None

        except Exception as e:
            logger.error(
                'Failed to cleanup LiteLLM team (resources may be orphaned)',
                extra={
                    'org_id': str(org_id),
                    'user_id': user_id,
                    'error': str(e),
                },
            )
            return e

    @staticmethod
    async def get_org_credits(user_id: str, org_id: UUID) -> float | None:
        """
        Get organization credits from LiteLLM team.

        Args:
            user_id: User ID
            org_id: Organization ID

        Returns:
            float | None: Credits (max_budget - spend) or None if LiteLLM not configured
        """
        try:
            user_team_info = await LiteLlmManager.get_user_team_info(
                user_id, str(org_id)
            )
            if not user_team_info:
                logger.warning(
                    'No team info available from LiteLLM',
                    extra={'user_id': user_id, 'org_id': str(org_id)},
                )
                return None

            max_budget = (user_team_info.get('litellm_budget_table') or {}).get(
                'max_budget', 0
            )
            spend = user_team_info.get('spend', 0)
            credits = max(max_budget - spend, 0)

            logger.debug(
                'Retrieved organization credits',
                extra={
                    'user_id': user_id,
                    'org_id': str(org_id),
                    'credits': credits,
                    'max_budget': max_budget,
                    'spend': spend,
                },
            )

            return credits

        except Exception as e:
            logger.warning(
                'Failed to retrieve organization credits',
                extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
            )
            return None
