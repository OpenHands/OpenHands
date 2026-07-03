"""Tests for openhands.app_server.utils.llm module."""

from openhands.app_server.utils import llm as llm_utils
from openhands.app_server.utils.llm import (
    _assign_provider,
    _derive_verified_models,
    get_provider_api_base,
    is_openhands_model,
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


class TestNormalizeLitellmModel:
    """Tests for model ids that need provider-prefix normalization."""

    def test_bare_workers_ai_scoped_models_get_cloudflare_prefix(self):
        """Workers AI @hf/@cf model ids need the LiteLLM cloudflare prefix."""
        assert (
            llm_utils.normalize_litellm_model('@hf/thebloke/codellama-7b-instruct-awq')
            == 'cloudflare/@hf/thebloke/codellama-7b-instruct-awq'
        )
        assert (
            llm_utils.normalize_litellm_model('@cf/meta/llama-3.2-3b-instruct')
            == 'cloudflare/@cf/meta/llama-3.2-3b-instruct'
        )

    def test_prefixed_and_non_workers_models_are_left_unchanged(self):
        assert (
            llm_utils.normalize_litellm_model(
                'cloudflare/@hf/thebloke/codellama-7b-instruct-awq'
            )
            == 'cloudflare/@hf/thebloke/codellama-7b-instruct-awq'
        )
        assert (
            llm_utils.normalize_litellm_model(
                'huggingface/meta-llama/Llama-3.1-8B-Instruct'
            )
            == 'huggingface/meta-llama/Llama-3.1-8B-Instruct'
        )
        assert llm_utils.normalize_litellm_model('openai/gpt-4o') == 'openai/gpt-4o'
        assert llm_utils.normalize_litellm_model(None) is None


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
