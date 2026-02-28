from typing import Annotated

from pydantic import BaseModel, EmailStr, Field, SecretStr, StringConstraints
from storage.org import Org
from storage.org_member import OrgMember
from storage.role import Role


class OrgCreationError(Exception):
    """Base exception for organization creation errors."""

    pass


class OrgNameExistsError(OrgCreationError):
    """Raised when an organization name already exists."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f'Organization with name "{name}" already exists')


class LiteLLMIntegrationError(OrgCreationError):
    """Raised when LiteLLM integration fails."""

    pass


class OrgDatabaseError(OrgCreationError):
    """Raised when database operations fail."""

    pass


class OrgDeletionError(Exception):
    """Base exception for organization deletion errors."""

    pass


class OrgAuthorizationError(OrgDeletionError):
    """Raised when user is not authorized to delete organization."""

    def __init__(self, message: str = 'Not authorized to delete organization'):
        super().__init__(message)


class OrphanedUserError(OrgDeletionError):
    """Raised when deleting an org would leave users without any organization."""

    def __init__(self, user_ids: list[str]):
        self.user_ids = user_ids
        super().__init__(
            f'Cannot delete organization: {len(user_ids)} user(s) would have no remaining organization'
        )


class OrgNotFoundError(Exception):
    """Raised when organization is not found or user doesn't have access."""

    def __init__(self, org_id: str):
        self.org_id = org_id
        super().__init__(f'Organization with id "{org_id}" not found')


class OrgMemberNotFoundError(Exception):
    """Raised when a member is not found in an organization."""

    def __init__(self, org_id: str, user_id: str):
        self.org_id = org_id
        self.user_id = user_id
        super().__init__(f'Member "{user_id}" not found in organization "{org_id}"')


class RoleNotFoundError(Exception):
    """Raised when a role is not found."""

    def __init__(self, role_id: int):
        self.role_id = role_id
        super().__init__(f'Role with id "{role_id}" not found')


class InvalidRoleError(Exception):
    """Raised when an invalid role name is specified."""

    def __init__(self, role_name: str):
        self.role_name = role_name
        super().__init__(f'Invalid role: "{role_name}"')


class InsufficientPermissionError(Exception):
    """Raised when user lacks permission to perform an operation."""

    def __init__(self, message: str = 'Insufficient permission'):
        super().__init__(message)


class CannotModifySelfError(Exception):
    """Raised when user attempts to modify their own membership."""

    def __init__(self, action: str = 'modify'):
        self.action = action
        super().__init__(f'Cannot {action} your own membership')


class LastOwnerError(Exception):
    """Raised when attempting to remove or demote the last owner."""

    def __init__(self, action: str = 'remove'):
        self.action = action
        super().__init__(f'Cannot {action} the last owner of an organization')


class MemberUpdateError(Exception):
    """Raised when member update operation fails."""

    def __init__(self, message: str = 'Failed to update member'):
        super().__init__(message)


class OrgCreate(BaseModel):
    """Request model for creating a new organization."""

    # Required fields
    name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
    ]
    contact_name: str
    contact_email: EmailStr


class OrgResponse(BaseModel):
    """Response model for organization."""

    id: str
    name: str
    contact_name: str
    contact_email: str
    conversation_expiration: int | None = None
    agent: str | None = None
    default_max_iterations: int | None = None
    security_analyzer: str | None = None
    confirmation_mode: bool | None = None
    default_llm_model: str | None = None
    default_llm_api_key_for_byor: str | None = None
    default_llm_base_url: str | None = None
    remote_runtime_resource_factor: int | None = None
    enable_default_condenser: bool = True
    billing_margin: float | None = None
    enable_proactive_conversation_starters: bool = True
    sandbox_base_container_image: str | None = None
    sandbox_runtime_container_image: str | None = None
    org_version: int = 0
    mcp_config: dict | None = None
    search_api_key: str | None = None
    sandbox_api_key: str | None = None
    max_budget_per_task: float | None = None
    enable_solvability_analysis: bool | None = None
    v1_enabled: bool | None = None
    credits: float | None = None
    is_personal: bool = False

    @classmethod
    def from_org(
        cls, org: Org, credits: float | None = None, user_id: str | None = None
    ) -> 'OrgResponse':
        """Create an OrgResponse from an Org entity.

        Args:
            org: The organization entity to convert
            credits: Optional credits value (defaults to None)
            user_id: Optional user ID to determine if org is personal (defaults to None)

        Returns:
            OrgResponse: The response model instance
        """
        return cls(
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
            default_llm_api_key_for_byor=None,
            default_llm_base_url=org.default_llm_base_url,
            remote_runtime_resource_factor=org.remote_runtime_resource_factor,
            enable_default_condenser=org.enable_default_condenser
            if org.enable_default_condenser is not None
            else True,
            billing_margin=org.billing_margin,
            enable_proactive_conversation_starters=org.enable_proactive_conversation_starters
            if org.enable_proactive_conversation_starters is not None
            else True,
            sandbox_base_container_image=org.sandbox_base_container_image,
            sandbox_runtime_container_image=org.sandbox_runtime_container_image,
            org_version=org.org_version if org.org_version is not None else 0,
            mcp_config=org.mcp_config,
            search_api_key=None,
            sandbox_api_key=None,
            max_budget_per_task=org.max_budget_per_task,
            enable_solvability_analysis=org.enable_solvability_analysis,
            v1_enabled=org.v1_enabled,
            credits=credits,
            is_personal=str(org.id) == user_id if user_id else False,
        )


class OrgPage(BaseModel):
    """Paginated response model for organization list."""

    items: list[OrgResponse]
    next_page_id: str | None = None
    current_org_id: str | None = None


class OrgUpdate(BaseModel):
    """Request model for updating an organization."""

    # Basic organization information (any authenticated user can update)
    name: Annotated[
        str | None,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
    ] = None
    contact_name: str | None = None
    contact_email: EmailStr | None = None
    conversation_expiration: int | None = None
    default_max_iterations: int | None = Field(default=None, gt=0)
    remote_runtime_resource_factor: int | None = Field(default=None, gt=0)
    billing_margin: float | None = Field(default=None, ge=0, le=1)
    enable_proactive_conversation_starters: bool | None = None
    sandbox_base_container_image: str | None = None
    sandbox_runtime_container_image: str | None = None
    mcp_config: dict | None = None
    sandbox_api_key: str | None = None
    max_budget_per_task: float | None = Field(default=None, gt=0)
    enable_solvability_analysis: bool | None = None
    v1_enabled: bool | None = None

    # LLM settings (require admin/owner role)
    default_llm_model: str | None = None
    default_llm_api_key_for_byor: str | None = None
    default_llm_base_url: str | None = None
    search_api_key: str | None = None
    security_analyzer: str | None = None
    agent: str | None = None
    confirmation_mode: bool | None = None
    enable_default_condenser: bool | None = None
    condenser_max_size: int | None = Field(default=None, ge=20)


class OrgMemberResponse(BaseModel):
    """Response model for a single organization member."""

    user_id: str
    email: str | None
    role_id: int
    role: str
    role_rank: int
    status: str | None


class OrgMemberPage(BaseModel):
    """Paginated response for organization members."""

    items: list[OrgMemberResponse]
    current_page: int = 1
    per_page: int = 10


class OrgMemberUpdate(BaseModel):
    """Request model for updating an organization member."""

    role: str | None = None  # Role name: 'owner', 'admin', or 'member'


class MeResponse(BaseModel):
    """Response model for the current user's membership in an organization."""

    org_id: str
    user_id: str
    email: str
    role: str
    llm_api_key: str
    max_iterations: int | None = None
    llm_model: str | None = None
    llm_api_key_for_byor: str | None = None
    llm_base_url: str | None = None
    status: str | None = None

    @staticmethod
    def _mask_key(secret: SecretStr | None) -> str:
        """Mask an API key, showing only last 4 characters."""
        if secret is None:
            return ''
        raw = secret.get_secret_value()
        if not raw:
            return ''
        if len(raw) <= 4:
            return '****'
        return '****' + raw[-4:]

    @classmethod
    def from_org_member(cls, member: OrgMember, role: Role, email: str) -> 'MeResponse':
        """Create a MeResponse from an OrgMember, Role, and user email.

        Args:
            member: The OrgMember entity
            role: The Role entity (provides role name)
            email: The user's email address

        Returns:
            MeResponse with masked API keys
        """
        return cls(
            org_id=str(member.org_id),
            user_id=str(member.user_id),
            email=email,
            role=role.name,
            llm_api_key=cls._mask_key(member.llm_api_key),
            max_iterations=member.max_iterations,
            llm_model=member.llm_model,
            llm_api_key_for_byor=cls._mask_key(member.llm_api_key_for_byor) or None,
            llm_base_url=member.llm_base_url,
            status=member.status,
        )
