import warnings

import httpx
from pydantic import BaseModel

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm
    from litellm import LlmProviders, ProviderConfigManager, get_llm_provider

from openhands.core.config import LLMConfig, OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm import bedrock

# ---------------------------------------------------------------------------
# The ``openhands-sdk`` package is the **single source of truth** for which
# models are verified and how bare LiteLLM names map to providers.
#
# Self-hosted mode builds the ``openhands/…`` model list from the SDK's
# ``VERIFIED_OPENHANDS_MODELS``.  SaaS mode overrides it with the database
# (via ``get_openhands_models``).
# ---------------------------------------------------------------------------
from openhands.sdk.llm.utils.verified_models import (  # noqa: E402
    VERIFIED_ANTHROPIC_MODELS as _SDK_ANTHROPIC,
)
from openhands.sdk.llm.utils.verified_models import (
    VERIFIED_MISTRAL_MODELS as _SDK_MISTRAL,
)
from openhands.sdk.llm.utils.verified_models import (
    VERIFIED_MODELS as _SDK_VERIFIED_MODELS,
)
from openhands.sdk.llm.utils.verified_models import (
    VERIFIED_OPENAI_MODELS as _SDK_OPENAI,
)
from openhands.sdk.llm.utils.verified_models import (
    VERIFIED_OPENHANDS_MODELS as _SDK_OPENHANDS,
)

# Build the ``openhands/…`` model list from the SDK.
OPENHANDS_MODELS: list[str] = [f'openhands/{m}' for m in _SDK_OPENHANDS]

CLARIFAI_MODELS = [
    'clarifai/openai.chat-completion.gpt-oss-120b',
    'clarifai/openai.chat-completion.gpt-oss-20b',
    'clarifai/openai.chat-completion.gpt-5',
    'clarifai/openai.chat-completion.gpt-5-mini',
    'clarifai/qwen.qwen3.qwen3-next-80B-A3B-Thinking',
    'clarifai/qwen.qwenLM.Qwen3-30B-A3B-Instruct-2507',
    'clarifai/qwen.qwenLM.Qwen3-30B-A3B-Thinking-2507',
    'clarifai/qwen.qwenLM.Qwen3-14B',
    'clarifai/qwen.qwenCoder.Qwen3-Coder-30B-A3B-Instruct',
    'clarifai/deepseek-ai.deepseek-chat.DeepSeek-R1-0528-Qwen3-8B',
    'clarifai/deepseek-ai.deepseek-chat.DeepSeek-V3_1',
    'clarifai/zai.completion.GLM_4_5',
    'clarifai/moonshotai.kimi.Kimi-K2-Instruct',
]

# ---------------------------------------------------------------------------
# Provider-assignment tables — derived from the SDK.
#
# LiteLLM returns some well-known models as *bare* names (e.g. ``gpt-5.2``
# instead of ``openai/gpt-5.2``).  The backend uses these sets to assign
# the canonical provider prefix *before* sending the list to the frontend.
# ---------------------------------------------------------------------------
VERIFIED_PROVIDERS: list[str] = list(_SDK_VERIFIED_MODELS.keys())

_BARE_OPENAI_MODELS: set[str] = set(_SDK_OPENAI)
_BARE_ANTHROPIC_MODELS: set[str] = set(_SDK_ANTHROPIC)
_BARE_MISTRAL_MODELS: set[str] = set(_SDK_MISTRAL)

DEFAULT_OPENHANDS_MODEL = 'openhands/claude-opus-4-5-20251101'


# ---------------------------------------------------------------------------
# Structured API response returned by ``/api/options/models``.
# ---------------------------------------------------------------------------
class ModelsResponse(BaseModel):
    """Structured response from the models endpoint.

    * ``models`` — flat list of ``provider/model`` strings (same shape as
      before, but bare names are now properly prefixed).
    * ``verified_models`` — model names (without provider prefix) that
      OpenHands has verified to work well.
    * ``verified_providers`` — provider names shown in the "Verified"
      section of the model selector.
    * ``default_model`` — the recommended default model id.
    """

    models: list[str]
    verified_models: list[str]
    verified_providers: list[str]
    default_model: str


def is_openhands_model(model: str | None) -> bool:
    """Check if the model uses the OpenHands provider.

    The SDK's ``AgentSettings`` validator automatically transforms
    ``openhands/X`` to ``litellm_proxy/X`` (the internal proxy name),
    so both prefixes must be recognised.

    Args:
        model: The model name to check.

    Returns:
        True if the model starts with 'openhands/' or 'litellm_proxy/',
        False otherwise.
    """
    return bool(
        model and (model.startswith('openhands/') or model.startswith('litellm_proxy/'))
    )


# Canonical masked placeholder for LLM API keys. Matches pydantic's
# ``SecretStr`` default representation so request/response payloads that pass
# through ``model_dump(mode='json')`` stay consistent with payloads that the
# enterprise org-settings validator constructs by hand. Importers should treat
# this as the single source of truth for "a key exists but its value is
# intentionally hidden."
MASKED_API_KEY = '**********'


def resolve_llm_base_url(
    model: str | None,
    base_url: str | None,
    *,
    managed_proxy_url: str,
) -> str | None:
    """Resolve the ``base_url`` to persist for an LLM configuration.

    Single source of truth for two code paths that otherwise duplicated the
    same logic:

    * ``openhands/app_server/settings/settings_router._post_merge_llm_fixups``
      (personal-settings save path).
    * ``enterprise/server/routes/org_models.OrgLLMSettingsUpdate._normalize_agent_settings``
      (org-defaults save path).

    Semantics:

    * ``base_url == ''`` → ``None`` (explicit "clear" signal from the UI;
      don't auto-infer on top of it).
    * ``base_url`` non-empty → returned unchanged.
    * ``base_url is None`` + known OpenHands / managed model → ``managed_proxy_url``.
    * ``base_url is None`` + known BYOR provider → default from
      :func:`get_provider_api_base`.
    * Any other combination → ``None``.

    Exceptions from ``litellm`` are logged and swallowed so a flaky provider
    lookup can never break a settings save.
    """
    if base_url == '':
        return None
    if base_url is not None:
        return base_url
    if not model:
        return None
    if is_openhands_model(model):
        return managed_proxy_url
    try:
        return get_provider_api_base(model)
    except Exception as e:
        logger.error(f'Failed to get api_base from litellm for model {model}: {e}')
        return None


def get_provider_api_base(model: str) -> str | None:
    """Get the API base URL for a model using litellm.

    This function tries multiple approaches to determine the API base URL:
    1. First tries litellm.get_api_base() which handles OpenAI, Gemini, Mistral
    2. Falls back to ProviderConfigManager.get_provider_model_info() for providers
       like Anthropic that have ModelInfo classes with get_api_base() methods

    Args:
        model: The model name (e.g., 'gpt-4', 'anthropic/claude-sonnet-4-5-20250929')

    Returns:
        The API base URL if found, None otherwise.
    """
    # First try get_api_base (handles OpenAI, Gemini with specific URL patterns)
    try:
        api_base = litellm.get_api_base(model, {})
        if api_base:
            return api_base
    except Exception:
        pass

    # Fall back to ProviderConfigManager for providers like Anthropic
    try:
        # Get the provider from the model
        _, provider_name, _, _ = get_llm_provider(model)
        if provider_name:
            # Convert provider name to LlmProviders enum
            try:
                provider_enum = LlmProviders(provider_name)
                model_info = ProviderConfigManager.get_provider_model_info(
                    model, provider_enum
                )
                if model_info and hasattr(model_info, 'get_api_base'):
                    return model_info.get_api_base()
            except ValueError:
                pass  # Provider not in enum
    except Exception:
        pass

    return None


def get_openhands_models(
    verified_models: list[str] | None = None,
) -> list[str]:
    """Return the list of OpenHands-provider model strings.

    In self-hosted mode *verified_models* is ``None`` (or empty) and the
    hardcoded ``OPENHANDS_MODELS`` list is used.  In SaaS mode the caller
    passes the database-backed list which takes precedence.

    Args:
        verified_models: Optional list of ``"openhands/<name>"`` strings
            loaded from the verified-models database table.

    Returns:
        A list such as ``["openhands/claude-opus-4-6", ...]``.
    """
    return verified_models if verified_models else OPENHANDS_MODELS


def _assign_provider(model: str) -> str:
    """Prefix a bare model name with its canonical provider.

    Models that already contain a ``/`` provider separator are returned
    unchanged. Only well-known bare names (OpenAI, Anthropic, Mistral)
    are prefixed.
    """
    if '/' in model:
        return model

    # Prefix well-known bare SDK model names with their canonical provider.
    # The provider sets are loaded from the SDK once at import time.
    if model in _BARE_OPENAI_MODELS:
        return f'openai/{model}'
    if model in _BARE_ANTHROPIC_MODELS:
        return f'anthropic/{model}'
    if model in _BARE_MISTRAL_MODELS:
        return f'mistral/{model}'
    return model


def _derive_verified_models(openhands_models: list[str]) -> list[str]:
    """Extract the bare model names from the ``openhands/…`` model list."""
    return [
        m.removeprefix('openhands/')
        for m in openhands_models
        if m.startswith('openhands/')
    ]


def get_supported_llm_models(
    config: OpenHandsConfig,
    verified_models: list[str] | None = None,
) -> ModelsResponse:
    """Collect every model available to this server and return structured data.

    The returned ``ModelsResponse`` contains:

    * a flat list of ``provider/model`` strings (bare LiteLLM names are
      prefixed with the correct provider),
    * a list of *verified* model names (the OpenHands-curated subset),
    * the set of verified providers, and
    * the recommended default model.

    Args:
        config: The OpenHands configuration.
        verified_models: Optional list of ``"openhands/<name>"`` strings
            from the database (SaaS mode).  When provided these replace the
            hardcoded ``OPENHANDS_MODELS``.
    """
    litellm_model_list = litellm.model_list + list(litellm.model_cost.keys())
    litellm_model_list_without_bedrock = bedrock.remove_error_modelId(
        litellm_model_list
    )
    # TODO: for bedrock, this is using the default config
    llm_config: LLMConfig = config.get_llm_config()
    bedrock_model_list: list[str] = []
    if (
        llm_config.aws_region_name
        and llm_config.aws_access_key_id
        and llm_config.aws_secret_access_key
    ):
        bedrock_model_list = bedrock.list_foundation_models(
            llm_config.aws_region_name,
            llm_config.aws_access_key_id.get_secret_value(),
            llm_config.aws_secret_access_key.get_secret_value(),
        )
    model_list = litellm_model_list_without_bedrock + bedrock_model_list
    for llm_config in config.llms.values():
        ollama_base_url = llm_config.ollama_base_url
        if llm_config.model.startswith('ollama'):
            if not ollama_base_url:
                ollama_base_url = llm_config.base_url
        if ollama_base_url:
            ollama_url = ollama_base_url.strip('/') + '/api/tags'
            try:
                ollama_models_list = httpx.get(ollama_url, timeout=3).json()['models']  # noqa: ASYNC100
                for model in ollama_models_list:
                    model_list.append('ollama/' + model['name'])
                break
            except httpx.HTTPError as e:
                logger.error(f'Error getting OLLAMA models: {e}')

    openhands_models = get_openhands_models(verified_models)

    # Assign canonical provider prefixes to bare LiteLLM names, then dedupe.
    all_models = (
        openhands_models + CLARIFAI_MODELS + [_assign_provider(m) for m in model_list]
    )
    unique_models = sorted(set(all_models))

    return ModelsResponse(
        models=unique_models,
        verified_models=_derive_verified_models(openhands_models),
        verified_providers=VERIFIED_PROVIDERS,
        default_model=DEFAULT_OPENHANDS_MODEL,
    )
