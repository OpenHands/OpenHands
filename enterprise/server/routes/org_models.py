from pydantic import BaseModel, EmailStr, Field


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


class OrgCreate(BaseModel):
    """Request model for creating a new organization."""

    # Required fields
    name: str = Field(min_length=1, max_length=255, strip_whitespace=True)
    contact_name: str
    contact_email: EmailStr = Field(strip_whitespace=True)


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
