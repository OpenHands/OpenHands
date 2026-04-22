from typing import Annotated, Any

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)
from server.constants import LITE_LLM_API_URL
from storage.org import Org
from storage.org_member import OrgMember
from storage.role import Role

from openhands.sdk.settings import AgentSettings, ConversationSettings
from openhands.utils.llm import MASKED_API_KEY, resolve_llm_base_url


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
    remote_runtime_resource_factor: int | None = None
    billing_margin: float | None = None
    enable_proactive_conversation_starters: bool = True
    sandbox_base_container_image: str | None = None
    sandbox_runtime_container_image: str | None = None
    org_version: int = 0
    agent_settings: AgentSettings = Field(default_factory=AgentSettings)
    conversation_settings: ConversationSettings = Field(
        default_factory=ConversationSettings
    )
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
        """Create an OrgResponse from an Org entity."""
        return cls(
            id=str(org.id),
            name=org.name,
            contact_name=org.contact_name,
            contact_email=org.contact_email,
            conversation_expiration=org.conversation_expiration,
            remote_runtime_resource_factor=org.remote_runtime_resource_factor,
            billing_margin=org.billing_margin,
            enable_proactive_conversation_starters=org.enable_proactive_conversation_starters
            if org.enable_proactive_conversation_starters is not None
            else True,
            sandbox_base_container_image=org.sandbox_base_container_image,
            sandbox_runtime_container_image=org.sandbox_runtime_container_image,
            org_version=org.org_version if org.org_version is not None else 0,
            agent_settings=AgentSettings.model_validate(
                dict(org.agent_settings) if org.agent_settings else {}
            ),
            conversation_settings=ConversationSettings.model_validate(
                dict(org.conversation_settings) if org.conversation_settings else {}
            ),
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
    """Request model for updating an organization.

    ``agent_settings_diff`` and ``conversation_settings_diff`` are sparse diffs
    that are deep-merged into the org row and then validated as full settings
    before persistence.
    """

    name: Annotated[
        str | None,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
    ] = None
    contact_name: str | None = None
    contact_email: EmailStr | None = None
    conversation_expiration: int | None = None
    remote_runtime_resource_factor: int | None = Field(default=None, gt=0)
    billing_margin: float | None = Field(default=None, ge=0, le=1)
    enable_proactive_conversation_starters: bool | None = None
    sandbox_base_container_image: str | None = None
    sandbox_runtime_container_image: str | None = None
    sandbox_api_key: str | None = None
    max_budget_per_task: float | None = Field(default=None, gt=0)
    enable_solvability_analysis: bool | None = None
    v1_enabled: bool | None = None
    search_api_key: str | None = None
    llm_api_key: str | None = None
    agent_settings_diff: dict[str, Any] | None = None
    conversation_settings_diff: dict[str, Any] | None = None

    @model_validator(mode='after')
    def _normalize_settings_diffs(self) -> 'OrgUpdate':
        """Normalize sparse settings diffs before merge/persistence."""
        self._normalize_agent_settings_diff()
        self._cleanup_empty_diff('agent_settings_diff', nested_key='llm')
        self._cleanup_empty_diff('conversation_settings_diff')
        return self

    def _normalize_agent_settings_diff(self) -> None:
        """Normalize nested LLM settings inside ``agent_settings_diff``."""
        llm_diff = self._get_agent_llm_diff()
        if llm_diff is None:
            return

        self._lift_and_mask_llm_api_key(llm_diff)
        self._resolve_agent_llm_base_url(llm_diff)

    def _get_agent_llm_diff(self) -> dict[str, Any] | None:
        """Return the nested ``llm`` diff when present and dictionary-shaped."""
        if self.agent_settings_diff is None:
            return None
        llm_diff = self.agent_settings_diff.get('llm')
        return llm_diff if isinstance(llm_diff, dict) else None

    def _lift_and_mask_llm_api_key(self, llm_diff: dict[str, Any]) -> None:
        """Lift nested api keys to ``llm_api_key`` and mask the JSON diff."""
        if 'api_key' not in llm_diff:
            return

        nested_key = llm_diff.pop('api_key')
        if (
            self.llm_api_key is None
            and nested_key is not None
            and nested_key != MASKED_API_KEY
        ):
            self.llm_api_key = nested_key
        if nested_key is not None:
            llm_diff['api_key'] = MASKED_API_KEY

    def _resolve_agent_llm_base_url(self, llm_diff: dict[str, Any]) -> None:
        """Fill provider-default base URLs for sparse LLM diffs when needed."""
        resolved_base_url = resolve_llm_base_url(
            model=llm_diff.get('model'),
            base_url=llm_diff.get('base_url'),
            managed_proxy_url=LITE_LLM_API_URL,
        )
        if resolved_base_url is not None:
            llm_diff['base_url'] = resolved_base_url

    def _cleanup_empty_diff(
        self,
        field_name: str,
        nested_key: str | None = None,
    ) -> None:
        """Drop empty nested diffs and collapse empty diff payloads to ``None``."""
        settings_diff = getattr(self, field_name)
        if not isinstance(settings_diff, dict):
            if not settings_diff:
                setattr(self, field_name, None)
            return

        if nested_key is not None and not settings_diff.get(nested_key):
            settings_diff.pop(nested_key, None)
        if not settings_diff:
            setattr(self, field_name, None)

    def updated_fields(self) -> set[str]:
        """Return the public field names explicitly present on the update."""
        return {
            field
            for field in type(self).model_fields
            if getattr(self, field) is not None
        }

    def has_updates(self) -> bool:
        """Check if any public update field is set (not None)."""
        return bool(self.updated_fields())

    def touches_org_defaults(self) -> bool:
        """Whether this update touches shared organization defaults."""
        return bool(
            self.updated_fields()
            & {
                'agent_settings_diff',
                'conversation_settings_diff',
                'search_api_key',
                'llm_api_key',
            }
        )

    def restricted_fields(self) -> set[str]:
        """Return fields that require elevated org settings permissions."""
        return self.updated_fields() & {
            'agent_settings_diff',
            'conversation_settings_diff',
            'search_api_key',
            'sandbox_api_key',
            'llm_api_key',
        }

    def model_update_dict(self) -> dict[str, Any]:
        """Return JSON-serializable scalar fields for persistence."""
        return self.model_dump(
            mode='json',
            exclude_none=True,
            exclude={'agent_settings_diff', 'conversation_settings_diff'},
        )

    def apply_to_org(self, org: Org) -> None:
        """Apply non-settings fields directly to the organization model."""
        for key, value in self.model_update_dict().items():
            if hasattr(org, key):
                setattr(org, key, value)

    def get_member_updates(self) -> 'OrgMemberSettingsUpdate | None':
        """Get shared updates that need to be propagated to org members.

        An empty ``llm_api_key`` means the org-wide custom key is being cleared
        (e.g. owner switching to a managed/OpenHands provider). It must not
        land in member rows — ``OrgMember.llm_api_key``'s setter has no
        ``if raw else None`` guard because the column is ``nullable=False``,
        so an empty string would become an encrypted empty blob rather than a
        cleared value. Coerce ``""`` to ``None`` so member rows are untouched.
        """
        member_settings = OrgMemberSettingsUpdate(
            agent_settings_diff=self.agent_settings_diff,
            conversation_settings_diff=self.conversation_settings_diff,
            llm_api_key=self.llm_api_key or None,
        )
        return member_settings if member_settings.has_updates() else None


class OrgDefaultsSettingsResponse(BaseModel):
    """Response model for organization default settings."""

    agent_settings: AgentSettings = Field(default_factory=AgentSettings)
    conversation_settings: ConversationSettings = Field(
        default_factory=ConversationSettings
    )
    llm_api_key_set: bool = False
    search_api_key: str | None = None  # Masked in response

    @staticmethod
    def _mask_key(secret: SecretStr | None) -> str | None:
        """Mask an API key, showing only last 4 characters."""
        if secret is None:
            return None
        raw = secret.get_secret_value()
        if not raw:
            return None
        if len(raw) <= 4:
            return '****'
        return '****' + raw[-4:]

    @classmethod
    def from_org(cls, org: Org) -> 'OrgDefaultsSettingsResponse':
        """Create response from Org entity.

        Denormalizes the SDK's ``litellm_proxy/`` prefix back to
        ``openhands/`` so the frontend's basic-view provider/model dropdowns
        can be populated, and nulls ``api_key`` so neither the raw secret
        nor the ``MASKED_API_KEY`` marker leaks in the response.
        ``base_url`` is returned exactly as stored so ``org.agent_settings``,
        ``org_member.agent_settings_diff`` and this response always carry
        the same value.
        """
        agent_settings = AgentSettings.model_validate(
            dict(org.agent_settings) if org.agent_settings else {}
        )
        cls._denormalize_llm_for_response(agent_settings)
        return cls(
            agent_settings=agent_settings,
            conversation_settings=ConversationSettings.model_validate(
                dict(org.conversation_settings) if org.conversation_settings else {}
            ),
            llm_api_key_set=org.llm_api_key is not None,
            search_api_key=cls._mask_key(org.search_api_key),
        )

    @staticmethod
    def _denormalize_llm_for_response(agent_settings: AgentSettings) -> None:
        """Rewrite ``agent_settings.llm`` in-place for UI consumption.

        * ``litellm_proxy/X`` → ``openhands/X`` so the basic-view provider
          dropdown matches (the SDK's ``AgentSettings`` validator
          normalizes the other direction on load).
        * ``base_url`` is returned **as stored** so the three sync targets
          (``org.agent_settings.llm.base_url``,
          ``org_member.agent_settings_diff.llm.base_url``, and the GET
          response) always agree. The frontend is responsible for
          recognizing the managed LiteLLM proxy URL / provider-default URL
          as "basic mode" — see ``KNOWN_PROVIDER_DEFAULT_BASE_URLS`` in
          ``frontend/src/routes/llm-settings.tsx``.
        * ``api_key`` is nulled so neither the raw secret nor the
          ``MASKED_API_KEY`` marker leaks in the response — the frontend
          reads ``llm_api_key_set`` to know whether a key exists.

        Pydantic v2 field assignment bypasses ``field_validator`` /
        ``model_validator`` by default (``validate_assignment`` is off on
        the SDK's ``LLM`` model), so the rename survives without being
        re-normalized back to ``litellm_proxy/``.
        """
        llm = agent_settings.llm
        if llm.model and llm.model.startswith('litellm_proxy/'):
            llm.model = f'openhands/{llm.model.removeprefix("litellm_proxy/")}'
        llm.api_key = None


class OrgMemberSettingsUpdate(BaseModel):
    """Shared settings updates that may be propagated to organization members.

    ``llm_api_key`` is typed as ``SecretStr`` so the raw value never ends up
    in logs or ``model_dump(mode='json')`` output by accident — the
    column-backed ``OrgMember.llm_api_key`` setter accepts ``SecretStr``
    directly and unwraps via ``get_secret_value()``.

    ``has_custom_llm_api_key`` propagates through
    ``update_all_members_settings_async`` so an org-defaults save can
    reset every member's "I have a personal BYOR key" flag in one pass —
    managed-mode switches rely on this to stop load-time fallthrough from
    returning stale custom markers.
    """

    agent_settings_diff: dict[str, Any] | None = None
    conversation_settings_diff: dict[str, Any] | None = None
    llm_api_key: SecretStr | None = None
    has_custom_llm_api_key: bool | None = None

    def has_updates(self) -> bool:
        """Check if any field is set (not None)."""
        return any(
            getattr(self, field) is not None for field in type(self).model_fields
        )


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
    """Response model for the current user's membership in an organization.

    ``agent_settings_diff`` and ``conversation_settings_diff`` carry the
    member-level overrides on top of the organization defaults.
    """

    org_id: str
    user_id: str
    email: str
    role: str
    llm_api_key: str
    llm_api_key_for_byor: str | None = None
    agent_settings_diff: dict[str, Any] = Field(default_factory=dict)
    conversation_settings_diff: dict[str, Any] = Field(default_factory=dict)
    status: str | None = None

    @staticmethod
    def _mask_key(secret: str | SecretStr | None) -> str:
        """Mask an API key, showing only last 4 characters."""
        if secret is None:
            return ''
        raw = secret.get_secret_value() if isinstance(secret, SecretStr) else secret
        if not raw:
            return ''
        if len(raw) <= 4:
            return '****'
        return '****' + raw[-4:]

    @classmethod
    def from_org_member(
        cls,
        member: OrgMember,
        role: Role,
        email: str,
    ) -> 'MeResponse':
        """Create a MeResponse from an OrgMember, Role, and user email."""
        return cls(
            org_id=str(member.org_id),
            user_id=str(member.user_id),
            email=email,
            role=role.name,
            llm_api_key=cls._mask_key(member.llm_api_key),
            llm_api_key_for_byor=cls._mask_key(member.llm_api_key_for_byor) or None,
            agent_settings_diff=dict(member.agent_settings_diff or {}),
            conversation_settings_diff=dict(member.conversation_settings_diff or {}),
            status=member.status,
        )


class OrgAppSettingsResponse(BaseModel):
    """Response model for organization app settings."""

    enable_proactive_conversation_starters: bool = True
    enable_solvability_analysis: bool | None = None
    max_budget_per_task: float | None = None

    @classmethod
    def from_org(cls, org: Org) -> 'OrgAppSettingsResponse':
        """Create an OrgAppSettingsResponse from an Org entity.

        Args:
            org: The organization entity

        Returns:
            OrgAppSettingsResponse with app settings
        """
        return cls(
            enable_proactive_conversation_starters=org.enable_proactive_conversation_starters
            if org.enable_proactive_conversation_starters is not None
            else True,
            enable_solvability_analysis=org.enable_solvability_analysis,
            max_budget_per_task=org.max_budget_per_task,
        )


class OrgAppSettingsUpdate(BaseModel):
    """Request model for updating organization app settings."""

    enable_proactive_conversation_starters: bool | None = None
    enable_solvability_analysis: bool | None = None
    max_budget_per_task: float | None = None

    @field_validator('max_budget_per_task')
    @classmethod
    def validate_max_budget_per_task(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError('max_budget_per_task must be greater than 0')
        return v


VALID_GIT_PROVIDERS = {'github', 'gitlab', 'bitbucket'}


class GitOrgClaimRequest(BaseModel):
    """Request model for claiming a Git organization."""

    provider: str
    git_organization: str

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_GIT_PROVIDERS:
            raise ValueError(
                f'Invalid provider: "{v}". Must be one of: {", ".join(sorted(VALID_GIT_PROVIDERS))}'
            )
        return v

    @field_validator('git_organization')
    @classmethod
    def validate_git_organization(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError('git_organization must not be empty')
        return v


class GitOrgClaimResponse(BaseModel):
    """Response model for a Git organization claim."""

    id: str
    org_id: str
    provider: str
    git_organization: str
    claimed_by: str
    claimed_at: str


class GitOrgAlreadyClaimedError(Exception):
    """Raised when a Git organization is already claimed by another OpenHands org."""

    def __init__(self, provider: str, git_organization: str):
        self.provider = provider
        self.git_organization = git_organization
        super().__init__(
            f'Git organization "{git_organization}" on {provider} is already claimed by another organization'
        )


class OrgMemberFinancialResponse(BaseModel):
    """Financial data for a single organization member."""

    user_id: str
    email: str | None
    lifetime_spend: float  # Total amount spent (from LiteLLM)
    current_budget: float  # Remaining budget (max_budget - spend)
    max_budget: float | None  # Total allocated budget (None = unlimited)


class OrgMemberFinancialPage(BaseModel):
    """Paginated response for organization member financial data."""

    items: list[OrgMemberFinancialResponse]
    current_page: int = 1
    per_page: int = 10
    next_page_id: str | None = None
