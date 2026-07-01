from __future__ import annotations

from typing import Any, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    SerializationInfo,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from openhands.app_server.utils.llm import resolve_llm_base_url
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.llm import LLM


def has_real_api_key(api_key: Any) -> bool:
    """Return True iff ``api_key`` carries a non-empty value.

    A ``SecretStr('')`` should report as *not set* — otherwise the UI tells
    the user a key is stored when it isn't. Mirrors the check used in
    ``Settings.llm_api_key_is_set``.
    """
    if api_key is None:
        return False
    secret_value = (
        api_key.get_secret_value() if isinstance(api_key, SecretStr) else str(api_key)
    )
    return bool(secret_value and secret_value.strip())


def resolve_profile_llm(
    profile_llm: LLM,
    *,
    managed_proxy_url: str,
    fallback_api_key: Any = None,
) -> LLM:
    """Resolve a saved profile's LLM for activation on the agent server.

    Fills the provider-default ``base_url`` when the profile saved none, and
    falls back to ``fallback_api_key`` (the user's effective settings key) when
    the profile carries no real key. Managed profiles persist a masked key, so
    without the fallback the agent server would call the LiteLLM proxy with no
    credentials; BYOR profiles keep their own key (the fallback is skipped).
    """
    resolved = profile_llm.model_copy(
        update={
            'base_url': resolve_llm_base_url(
                model=profile_llm.model,
                base_url=profile_llm.base_url,
                managed_proxy_url=managed_proxy_url,
            ),
            # Force streaming on (profiles default to the SDK's stream=False).
            'stream': True,
        }
    )
    if not has_real_api_key(resolved.api_key) and has_real_api_key(fallback_api_key):
        resolved = resolved.model_copy(update={'api_key': fallback_api_key})
    return resolved


# Soft cap — keeps Settings payload bounded and blocks per-user storage
# blow-ups. Tune if product requirements change.
MAX_PROFILES_PER_USER: Final[int] = 10


class ProfileNotFoundError(LookupError):
    """Raised when a profile lookup or activation references an unknown name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Profile '{name}' not found")


class ProfileLimitExceededError(ValueError):
    """Raised when saving a new profile would exceed :data:`MAX_PROFILES_PER_USER`."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(
            f'Profile limit reached ({limit}). Delete a profile before saving a new one.'
        )


class ProfileAlreadyExistsError(ValueError):
    """Raised when a rename target collides with an existing profile."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Profile '{name}' already exists")


class StrictLLM(LLM):
    """LLM variant that rejects unknown fields.

    The base ``LLM`` model has ``extra='ignore'``, so typos and renamed keys
    silently disappear. For API input we want to fail loud, otherwise users
    can POST ``{"llm": {"custom_header": "x"}}`` and get a 201 with the
    field quietly dropped.
    """

    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='wrap')
    @classmethod
    def _restore_is_subscription(cls, data: Any, handler: Any) -> Any:
        """Temporary workaround for a non-composable SDK validator.

        ``LLM`` defines a validator of this same name (``mode="wrap"``)
        that restores ``_is_subscription`` from this computed field so it
        survives a dump/validate round trip. It does so by reading its own
        raw ``data`` argument directly, independent of any subclass
        ``mode="before"`` validator's transformation of the input — so a
        normal ``before`` validator can't neutralize it, and the *only*
        way to override the behavior is to shadow this exact method name.
        See OpenHands/software-agent-sdk#3942.

        Strip ``is_subscription`` from the input before validation (so
        ``extra='forbid'`` doesn't reject the GET-response echo as an
        unrecognized field on this endpoint's GET-edit-POST round trip)
        and don't restore it: it's semantically only ever supposed to be
        set via ``LLM.subscription_login()``, never via user-supplied
        JSON here.

        TODO: remove once software-agent-sdk#3942 is fixed upstream.
        """
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != 'is_subscription'}
        return handler(data)


class LLMProfiles(BaseModel):
    """Container for saved LLM configurations.

    Stores a named collection of ``LLM`` configurations plus the name of the
    currently active one (if any). All profile-management logic lives here;
    ``Settings`` holds a single ``LLMProfiles`` instance and delegates to it.

    Invariants (enforced on validate + assignment):
    - ``active`` is either ``None`` or a key of ``profiles``.
    - Individual profiles that fail to parse (schema drift) are dropped with
      a warning rather than failing the whole ``Settings`` load.
    """

    model_config = ConfigDict(validate_assignment=True)

    profiles: dict[str, LLM] = Field(default_factory=dict)
    active: str | None = None

    # ── Validation ─────────────────────────────────────────────────

    @field_validator('profiles', mode='before')
    @classmethod
    def _skip_invalid_profiles(cls, value: Any) -> Any:
        """Best-effort per-profile load: skip entries that fail to validate.

        Guards against schema drift — if a single stored profile becomes
        invalid after an LLM-model upgrade, the user's other profiles and
        the rest of their settings still load.
        """
        if not isinstance(value, dict):
            return value
        valid: dict[str, Any] = {}
        for name, raw in value.items():
            if isinstance(raw, LLM):
                valid[name] = raw
                continue
            try:
                valid[name] = LLM.model_validate(raw)
            except ValidationError as exc:
                logger.warning('Skipping invalid LLM profile %r: %s', name, exc)
        return valid

    @model_validator(mode='after')
    def _reconcile_active(self) -> LLMProfiles:
        if self.active is not None and self.active not in self.profiles:
            # Bypass validate_assignment to avoid re-entering this validator.
            object.__setattr__(self, 'active', None)
        return self

    # ── Queries ────────────────────────────────────────────────────

    def get(self, name: str) -> LLM | None:
        """Return the profile's LLM or ``None`` if it doesn't exist."""
        return self.profiles.get(name)

    def require(self, name: str) -> LLM:
        """Return the profile's LLM or raise :class:`ProfileNotFoundError`."""
        llm = self.profiles.get(name)
        if llm is None:
            raise ProfileNotFoundError(name)
        return llm

    def has(self, name: str) -> bool:
        return name in self.profiles

    def summaries(
        self, *, managed_proxy_url: str | None = None
    ) -> list[dict[str, Any]]:
        """Return a ``{name, model, base_url, api_key_set}`` dict per profile.

        ``api_key_set`` mirrors the ``llm_api_key_set`` convention the main
        settings endpoint already uses, so the frontend can render
        "key stored" vs. "needs key" without fetching each profile.

        When ``managed_proxy_url`` is provided, ``base_url`` is resolved to the
        value the profile will actually use at runtime for public OpenHands
        provider profiles.
        """
        return [
            {
                'name': name,
                'model': llm.model,
                'base_url': (
                    resolve_llm_base_url(
                        llm.model, llm.base_url, managed_proxy_url=managed_proxy_url
                    )
                    if managed_proxy_url is not None
                    else llm.base_url
                ),
                'api_key_set': has_real_api_key(llm.api_key),
            }
            for name, llm in self.profiles.items()
        ]

    # ── Mutations ──────────────────────────────────────────────────

    def save(self, name: str, llm: LLM, include_secrets: bool = True) -> None:
        """Save ``llm`` under ``name``. Overwrites if the name exists.

        Always stores a copy so later caller-side mutations do not bleed into
        the stored profile. Raises :class:`ProfileLimitExceededError` if
        saving a *new* profile would push the count past
        :data:`MAX_PROFILES_PER_USER`.
        """
        if name not in self.profiles and len(self.profiles) >= MAX_PROFILES_PER_USER:
            raise ProfileLimitExceededError(MAX_PROFILES_PER_USER)

        update = {} if include_secrets else {'api_key': None}
        self.profiles[name] = llm.model_copy(update=update)

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename a profile, preserving stored LLM config, insertion order, and
        the active flag (if the renamed profile was active).

        Raises :class:`ProfileNotFoundError` if ``old_name`` doesn't exist,
        or :class:`ProfileAlreadyExistsError` if ``new_name`` is already taken
        by a different profile.
        """
        if old_name not in self.profiles:
            raise ProfileNotFoundError(old_name)
        if new_name == old_name:
            return
        if new_name in self.profiles:
            raise ProfileAlreadyExistsError(new_name)

        # Capture the active name *before* reassigning ``profiles`` — the
        # model_validator runs on assignment and would null out ``active``
        # (old_name no longer exists in the rebuilt dict), so we'd lose the
        # signal otherwise.
        was_active = self.active == old_name

        # Rebuild to preserve insertion order — the renamed profile keeps
        # the slot of the old one rather than moving to the end.
        renamed: dict[str, LLM] = {
            (new_name if key == old_name else key): llm
            for key, llm in self.profiles.items()
        }
        self.profiles = renamed
        if was_active:
            # Bypass validate_assignment since we know the invariant holds
            # (new_name is now a key of self.profiles).
            object.__setattr__(self, 'active', new_name)

    def delete(self, name: str) -> bool:
        """Delete a profile. Returns True if the profile existed.

        Clears ``active`` if the deleted profile was active.
        """
        if name not in self.profiles:
            return False
        del self.profiles[name]
        if self.active == name:
            # Bypass validate_assignment since we already know the invariant holds.
            object.__setattr__(self, 'active', None)
        return True

    # ── Serialization ──────────────────────────────────────────────

    @field_serializer('profiles')
    def _profiles_serializer(
        self,
        profiles: dict[str, LLM],
        info: SerializationInfo,
    ) -> dict[str, Any]:
        return {
            name: llm.model_dump(mode='json', context=info.context)
            for name, llm in profiles.items()
        }
