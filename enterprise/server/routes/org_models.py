from typing import Annotated

from pydantic import BaseModel, EmailStr, Field, StringConstraints
from storage.org import Org


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


class OrgNotFoundError(Exception):
    """Raised when organization is not found or user doesn't have access."""

    def __init__(self, org_id: str):
        self.org_id = org_id
        super().__init__(f'Organization with id "{org_id}" not found')


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

    @classmethod
    def from_org(cls, org: Org, credits: float | None = None) -> 'OrgResponse':
        """Create an OrgResponse from an Org entity.

        Args:
            org: The organization entity to convert
            credits: Optional credits value (defaults to None)

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
        )


class OrgPage(BaseModel):
    """Paginated response model for organization list."""

    items: list[OrgResponse]
    next_page_id: str | None = None


class OrgUpdate(BaseModel):
    """Request model for updating an organization."""

    # Basic organization information (any authenticated user can update)
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
