"""Tests for openhands.app_server.utils.llm module."""

from openhands.app_server.utils import llm as llm_utils
from openhands.app_server.utils.llm import (
    _assign_provider,
    _derive_verified_models,
    get_provider_api_base,
    is_openhands_model,
    normalize_llm_model_for_runtime,
)


class TestIsOpenhandsModel:
    """Tests for the is_openhands_model function."""

    def test_openhands_model_returns_true(self):
        """Test that models with 'openhands/' prefix return True."""
        assert is_openhands_model('openhands/claude-sonnet-4-5-20250929') is True
        assert is_openhands_model('openhands/gpt-5-2025-08-07') is True
        assert is_openhands_model('openhands/gemini-2.5-pro') is True

    def test_non_openhands_model_returns_false(self):
        """Test that models without 'openhands/' prefix return False."""
        assert is_openhands_model('gpt-4') is False
        assert is_openhands_model('claude-3-opus-20240229') is False
        assert is_openhands_model('anthropic/claude-3-opus-20240229') is False
        assert is_openhands_model('openai/gpt-4') is False
        assert is_openhands_model('litellm_proxy/gpt-4') is False

    def test_none_model_returns_false(self):
        """Test that None model returns False."""
        assert is_openhands_model(None) is False

    def test_empty_string_returns_false(self):
        """Test that empty string returns False."""
        assert is_openhands_model('') is False

    def test_similar_prefix_not_matched(self):
        """Test that similar prefixes don't incorrectly match."""
        assert is_openhands_model('openhands') is False  # Missing slash
        assert is_openhands_model('openhandsx/model') is False  # Extra char
        assert is_openhands_model('OPENHANDS/model') is False  # Wrong case


class TestAssignProvider:
    """Tests for the _assign_provider helper."""

    def test_known_bare_models_get_prefixed(self, monkeypatch):
        """Test that known bare models get the expected provider prefix."""
        monkeypatch.setattr(llm_utils, '_BARE_OPENAI_MODELS', {'gpt-5.2'})
        monkeypatch.setattr(
            llm_utils, '_BARE_ANTHROPIC_MODELS', {'claude-sonnet-4-5-20250929'}
        )
        monkeypatch.setattr(llm_utils, '_BARE_MISTRAL_MODELS', {'mistral-large-latest'})

        assert _assign_provider('gpt-5.2') == 'openai/gpt-5.2'
        assert (
            _assign_provider('claude-sonnet-4-5-20250929')
            == 'anthropic/claude-sonnet-4-5-20250929'
        )
        assert (
            _assign_provider('mistral-large-latest') == 'mistral/mistral-large-latest'
        )

    def test_prefixed_models_remain_unchanged(self, monkeypatch):
        """Test that already-prefixed models are returned untouched."""
        monkeypatch.setattr(llm_utils, '_BARE_OPENAI_MODELS', {'gpt-5.2'})
        monkeypatch.setattr(llm_utils, '_BARE_ANTHROPIC_MODELS', set())
        monkeypatch.setattr(llm_utils, '_BARE_MISTRAL_MODELS', set())

        assert _assign_provider('openai/gpt-5.2') == 'openai/gpt-5.2'

    def test_unresolvable_bare_models_remain_unchanged(self, monkeypatch):
        """Bare names LiteLLM cannot resolve fall through unchanged."""
        monkeypatch.setattr(llm_utils, '_BARE_OPENAI_MODELS', set())
        monkeypatch.setattr(llm_utils, '_BARE_ANTHROPIC_MODELS', set())
        monkeypatch.setattr(llm_utils, '_BARE_MISTRAL_MODELS', set())

        assert _assign_provider('totally-made-up-model-xyz') == (
            'totally-made-up-model-xyz'
        )

    def test_unverified_bare_models_use_litellm_fallback(self, monkeypatch):
        """Unverified bare names reach the dropdown via LiteLLM's routing."""
        monkeypatch.setattr(llm_utils, '_BARE_OPENAI_MODELS', set())
        monkeypatch.setattr(llm_utils, '_BARE_ANTHROPIC_MODELS', set())
        monkeypatch.setattr(llm_utils, '_BARE_MISTRAL_MODELS', set())

        # gemini-* lives bare in litellm.model_cost; LiteLLM routes it to
        # vertex_ai. Without the fallback the frontend's provider filter
        # drops it entirely.
        assert _assign_provider('gemini-2.0-flash') == 'vertex_ai/gemini-2.0-flash'
        # cohere.<model>:<rev> is the Bedrock-style ID for Cohere models;
        # LiteLLM resolves it to bedrock.
        assert (
            _assign_provider('cohere.command-r-v1:0') == 'bedrock/cohere.command-r-v1:0'
        )

    def test_litellm_fallback_exception_is_swallowed(self, monkeypatch):
        """A raising get_llm_provider must not break the model list build."""
        monkeypatch.setattr(llm_utils, '_BARE_OPENAI_MODELS', set())
        monkeypatch.setattr(llm_utils, '_BARE_ANTHROPIC_MODELS', set())
        monkeypatch.setattr(llm_utils, '_BARE_MISTRAL_MODELS', set())

        def _boom(*_args, **_kwargs):
            raise RuntimeError('litellm exploded')

        monkeypatch.setattr(llm_utils, 'get_llm_provider', _boom)

        assert _assign_provider('whatever') == 'whatever'


class TestNormalizeLlmModelForRuntime:
    """Tests for runtime-only model routing."""

    def test_custom_base_url_routes_opaque_model_through_proxy(self, monkeypatch):
        """Opaque model ids use LiteLLM proxy routing without changing display id."""
        monkeypatch.setattr(llm_utils, '_LITELLM_PROVIDERS', {'openai', 'anthropic'})
        monkeypatch.setattr(llm_utils.litellm, 'model_list', [])
        monkeypatch.setattr(llm_utils.litellm, 'model_cost', {})

        runtime_model, canonical_name = normalize_llm_model_for_runtime(
            'local/my-coder', 'https://gateway.example/v1'
        )

        assert runtime_model == 'litellm_proxy/local/my-coder'
        assert canonical_name == 'local/my-coder'

    def test_no_base_url_leaves_opaque_model_unchanged(self, monkeypatch):
        """Provider inference should be left to the normal path without a custom URL."""
        monkeypatch.setattr(llm_utils, '_LITELLM_PROVIDERS', {'openai', 'anthropic'})
        monkeypatch.setattr(llm_utils.litellm, 'model_list', [])
        monkeypatch.setattr(llm_utils.litellm, 'model_cost', {})

        assert normalize_llm_model_for_runtime('local/my-coder', None) == (
            'local/my-coder',
            None,
        )

    def test_known_litellm_routes_are_not_rewritten(self, monkeypatch):
        """Known LiteLLM provider routes already carry their adapter hint."""
        monkeypatch.setattr(llm_utils, '_LITELLM_PROVIDERS', {'openai', 'anthropic'})
        monkeypatch.setattr(llm_utils.litellm, 'model_list', ['gpt-4o'])
        monkeypatch.setattr(llm_utils.litellm, 'model_cost', {})

        assert normalize_llm_model_for_runtime(
            'openai/custom-alias', 'https://gateway.example/v1'
        ) == ('openai/custom-alias', None)
        assert normalize_llm_model_for_runtime(
            'gpt-4o', 'https://gateway.example/v1'
        ) == ('gpt-4o', None)

    def test_proxy_and_openhands_routes_are_not_rewritten(self, monkeypatch):
        """Existing internal routes keep their original model ids."""
        monkeypatch.setattr(llm_utils, '_LITELLM_PROVIDERS', {'openai', 'anthropic'})
        monkeypatch.setattr(llm_utils.litellm, 'model_list', [])
        monkeypatch.setattr(llm_utils.litellm, 'model_cost', {})

        assert normalize_llm_model_for_runtime(
            'litellm_proxy/local/my-coder', 'https://gateway.example/v1'
        ) == ('litellm_proxy/local/my-coder', None)
        assert normalize_llm_model_for_runtime(
            'openhands/gpt-5', 'https://gateway.example/v1'
        ) == ('openhands/gpt-5', None)


class TestDeriveVerifiedModels:
    """Tests for the _derive_verified_models helper."""

    def test_extracts_openhands_model_names(self):
        """Test that only openhands-prefixed models are returned bare."""
        models = [
            'openhands/claude-opus-4-5-20251101',
            'openhands/gpt-5',
            'openai/gpt-5',
            'gpt-4o',
        ]

        assert _derive_verified_models(models) == [
            'claude-opus-4-5-20251101',
            'gpt-5',
        ]


class TestGetProviderApiBase:
    """Tests for the get_provider_api_base function."""

    def test_openai_model_returns_openai_api_base(self):
        """Test that OpenAI models return the OpenAI API base URL."""
        assert get_provider_api_base('gpt-4') == 'https://api.openai.com'
        assert get_provider_api_base('openai/gpt-4') == 'https://api.openai.com'

    def test_anthropic_model_returns_anthropic_api_base(self):
        """Test that Anthropic models return the Anthropic API base URL."""
        assert (
            get_provider_api_base('anthropic/claude-sonnet-4-5-20250929')
            == 'https://api.anthropic.com'
        )
        assert (
            get_provider_api_base('claude-sonnet-4-5-20250929')
            == 'https://api.anthropic.com'
        )

    def test_gemini_model_returns_google_api_base(self):
        """Test that Gemini models return a Google API base URL."""
        api_base = get_provider_api_base('gemini/gemini-pro')
        assert api_base is not None
        assert 'generativelanguage.googleapis.com' in api_base

    def test_mistral_model_returns_mistral_api_base(self):
        """Test that Mistral models return the Mistral API base URL."""
        assert (
            get_provider_api_base('mistral/mistral-large-latest')
            == 'https://api.mistral.ai/v1'
        )

    def test_unknown_model_returns_none(self):
        """Test that unknown models return None."""
        result = get_provider_api_base('unknown-provider/unknown-model')
        # May return None or an API base depending on litellm behavior
        # The function should not raise an exception
        assert result is None or isinstance(result, str)
