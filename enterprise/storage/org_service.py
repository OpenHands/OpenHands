"""
Service class for managing organization operations.
Separates business logic from route handlers.
"""

from uuid import UUID, uuid4
from uuid import UUID as parse_uuid

from server.constants import ORG_SETTINGS_VERSION, get_default_litellm_model
from server.routes.org_models import (
    LiteLLMIntegrationError,
    OrgAuthorizationError,
    OrgDatabaseError,
    OrgNameExistsError,
    OrgNotFoundError,
    OrgUpdate,
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
    def has_admin_or_owner_role(user_id: str, org_id: UUID) -> bool:
        """
        Check if user has admin or owner role in the specified organization.

        Args:
            user_id: User ID to check
            org_id: Organization ID to check membership in

        Returns:
            bool: True if user has admin or owner role, False otherwise
        """
        try:
            # Parse user_id as UUID for database query
            user_uuid = parse_uuid(user_id)

            # Get the user's membership in this organization
            # Note: The type annotation says int but the actual column is UUID
            org_member = OrgMemberStore.get_org_member(org_id, user_uuid)
            if not org_member:
                return False

            # Get the role details
            role = RoleStore.get_role_by_id(org_member.role_id)
            if not role:
                return False

            # Admin and owner roles have elevated permissions
            # Based on test files, both admin and owner have rank 1
            return role.name in ['admin', 'owner']

        except Exception as e:
            logger.warning(
                'Error checking user role in organization',
                extra={
                    'user_id': user_id,
                    'org_id': str(org_id),
                    'error': str(e),
                },
            )
            return False

    @staticmethod
    def is_org_member(user_id: str, org_id: UUID) -> bool:
        """
        Check if user is a member of the specified organization.

        Args:
            user_id: User ID to check
            org_id: Organization ID to check membership in

        Returns:
            bool: True if user is a member, False otherwise
        """
        try:
            user_uuid = parse_uuid(user_id)
            org_member = OrgMemberStore.get_org_member(org_id, user_uuid)
            return org_member is not None
        except Exception as e:
            logger.warning(
                'Error checking user membership in organization',
                extra={
                    'user_id': user_id,
                    'org_id': str(org_id),
                    'error': str(e),
                },
            )
            return False

    @staticmethod
    def _get_llm_settings_fields() -> set[str]:
        """
        Get the set of organization fields that are considered LLM settings
        and require admin/owner role to update.

        Returns:
            set[str]: Set of field names that require elevated permissions
        """
        return {
            'default_llm_model',
            'default_llm_api_key_for_byor',
            'default_llm_base_url',
            'search_api_key',
            'security_analyzer',
            'agent',
            'confirmation_mode',
            'enable_default_condenser',
            'condenser_max_size',
        }

    @staticmethod
    def _has_llm_settings_updates(update_data: OrgUpdate) -> set[str]:
        """
        Check if the update contains any LLM settings fields.

        Args:
            update_data: The organization update data

        Returns:
            set[str]: Set of LLM fields being updated (empty if none)
        """
        llm_fields = OrgService._get_llm_settings_fields()
        update_dict = update_data.model_dump(exclude_none=True)
        return llm_fields.intersection(update_dict.keys())

    @staticmethod
    async def update_org_with_permissions(
        org_id: UUID,
        update_data: OrgUpdate,
        user_id: str,
    ) -> Org:
        """
        Update organization with permission checks for LLM settings.

        Args:
            org_id: Organization UUID to update
            update_data: Organization update data from request
            user_id: ID of the user requesting the update

        Returns:
            Org: The updated organization object

        Raises:
            ValueError: If organization not found
            PermissionError: If user is not a member, or lacks admin/owner role for LLM settings
            OrgDatabaseError: If database update fails
        """
        logger.info(
            'Updating organization with permission checks',
            extra={
                'org_id': str(org_id),
                'user_id': user_id,
                'has_update_data': update_data is not None,
            },
        )

        # Validate organization exists
        existing_org = OrgStore.get_org_by_id(org_id)
        if not existing_org:
            raise ValueError(f'Organization with ID {org_id} not found')

        # Check if user is a member of this organization
        if not OrgService.is_org_member(user_id, org_id):
            logger.warning(
                'Non-member attempted to update organization',
                extra={
                    'user_id': user_id,
                    'org_id': str(org_id),
                },
            )
            raise PermissionError(
                'User must be a member of the organization to update it'
            )

        # Check if update contains any LLM settings
        llm_fields_being_updated = OrgService._has_llm_settings_updates(update_data)
        if llm_fields_being_updated:
            # Verify user has admin or owner role
            has_permission = OrgService.has_admin_or_owner_role(user_id, org_id)
            if not has_permission:
                logger.warning(
                    'User attempted to update LLM settings without permission',
                    extra={
                        'user_id': user_id,
                        'org_id': str(org_id),
                        'attempted_fields': list(llm_fields_being_updated),
                    },
                )
                raise PermissionError(
                    'Admin or owner role required to update LLM settings'
                )

            logger.debug(
                'User has permission to update LLM settings',
                extra={
                    'user_id': user_id,
                    'org_id': str(org_id),
                    'llm_fields': list(llm_fields_being_updated),
                },
            )

        # Convert to dict for OrgStore (excluding None values)
        update_dict = update_data.model_dump(exclude_none=True)
        if not update_dict:
            logger.info(
                'No fields to update',
                extra={'org_id': str(org_id), 'user_id': user_id},
            )
            return existing_org

        # Perform the update
        try:
            updated_org = OrgStore.update_org(org_id, update_dict)
            if not updated_org:
                raise OrgDatabaseError('Failed to update organization in database')

            logger.info(
                'Organization updated successfully',
                extra={
                    'org_id': str(org_id),
                    'user_id': user_id,
                    'updated_fields': list(update_dict.keys()),
                },
            )

            return updated_org

        except Exception as e:
            logger.error(
                'Failed to update organization',
                extra={
                    'org_id': str(org_id),
                    'user_id': user_id,
                    'error': str(e),
                },
            )
            raise OrgDatabaseError(f'Failed to update organization: {str(e)}')

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

    @staticmethod
    def get_user_orgs_paginated(
        user_id: str, page_id: str | None = None, limit: int = 100
    ):
        """
        Get paginated list of organizations for a user.

        Args:
            user_id: User ID (string that will be converted to UUID)
            page_id: Optional page ID (offset as string) for pagination
            limit: Maximum number of organizations to return

        Returns:
            Tuple of (list of Org objects, next_page_id or None)
        """
        logger.debug(
            'Fetching paginated organizations for user',
            extra={'user_id': user_id, 'page_id': page_id, 'limit': limit},
        )

        # Convert user_id string to UUID
        user_uuid = parse_uuid(user_id)

        # Fetch organizations from store
        orgs, next_page_id = OrgStore.get_user_orgs_paginated(
            user_id=user_uuid, page_id=page_id, limit=limit
        )

        logger.debug(
            'Retrieved organizations for user',
            extra={
                'user_id': user_id,
                'org_count': len(orgs),
                'has_more': next_page_id is not None,
            },
        )

        return orgs, next_page_id

    @staticmethod
    async def get_org_by_id(org_id: UUID, user_id: str) -> Org:
        """
        Get organization by ID with membership validation.

        This method verifies that the user is a member of the organization
        before returning the organization details.

        Args:
            org_id: Organization ID
            user_id: User ID (string that will be converted to UUID)

        Returns:
            Org: The organization object

        Raises:
            OrgNotFoundError: If organization not found or user is not a member
        """
        logger.info(
            'Retrieving organization',
            extra={'user_id': user_id, 'org_id': str(org_id)},
        )

        # Verify user is a member of the organization
        org_member = OrgMemberStore.get_org_member(org_id, parse_uuid(user_id))
        if not org_member:
            logger.warning(
                'User is not a member of organization or organization does not exist',
                extra={'user_id': user_id, 'org_id': str(org_id)},
            )
            raise OrgNotFoundError(str(org_id))

        # Retrieve organization
        org = OrgStore.get_org_by_id(org_id)
        if not org:
            logger.error(
                'Organization not found despite valid membership',
                extra={'user_id': user_id, 'org_id': str(org_id)},
            )
            raise OrgNotFoundError(str(org_id))

        logger.info(
            'Successfully retrieved organization',
            extra={
                'org_id': str(org.id),
                'org_name': org.name,
                'user_id': user_id,
            },
        )

        return org

    @staticmethod
    def verify_owner_authorization(user_id: str, org_id: UUID) -> None:
        """
        Verify that the user is the owner of the organization.

        Args:
            user_id: User ID to check
            org_id: Organization ID

        Raises:
            OrgNotFoundError: If organization doesn't exist
            OrgAuthorizationError: If user is not authorized to delete
        """
        # Check if organization exists
        org = OrgStore.get_org_by_id(org_id)
        if not org:
            raise OrgNotFoundError(str(org_id))

        # Check if user is a member of the organization
        org_member = OrgMemberStore.get_org_member(org_id, parse_uuid(user_id))
        if not org_member:
            raise OrgAuthorizationError('User is not a member of this organization')

        # Check if user has owner role
        role = RoleStore.get_role_by_id(org_member.role_id)
        if not role or role.name != 'owner':
            raise OrgAuthorizationError(
                'Only organization owners can delete organizations'
            )

        logger.debug(
            'User authorization verified for organization deletion',
            extra={'user_id': user_id, 'org_id': str(org_id), 'role': role.name},
        )

    @staticmethod
    async def delete_org_with_cleanup(user_id: str, org_id: UUID) -> Org:
        """
        Delete organization with complete cleanup of all associated data.

        This method performs the complete organization deletion workflow:
        1. Verifies user authorization (owner only)
        2. Performs database cascade deletion and LiteLLM cleanup in single transaction

        Args:
            user_id: User ID requesting deletion (must be owner)
            org_id: Organization ID to delete

        Returns:
            Org: The deleted organization details

        Raises:
            OrgNotFoundError: If organization doesn't exist
            OrgAuthorizationError: If user is not authorized to delete
            OrgDatabaseError: If database operations or LiteLLM cleanup fail
        """
        logger.info(
            'Starting organization deletion',
            extra={'user_id': user_id, 'org_id': str(org_id)},
        )

        # Step 1: Verify user authorization
        OrgService.verify_owner_authorization(user_id, org_id)

        # Step 2: Perform database cascade deletion with LiteLLM cleanup in transaction
        try:
            deleted_org = await OrgStore.delete_org_cascade(org_id)
            if not deleted_org:
                # This shouldn't happen since we verified existence above
                raise OrgDatabaseError('Organization not found during deletion')

            logger.info(
                'Organization deletion completed successfully',
                extra={
                    'user_id': user_id,
                    'org_id': str(org_id),
                    'org_name': deleted_org.name,
                },
            )

            return deleted_org

        except Exception as e:
            logger.error(
                'Organization deletion failed',
                extra={'user_id': user_id, 'org_id': str(org_id), 'error': str(e)},
            )
            raise OrgDatabaseError(f'Failed to delete organization: {str(e)}')
