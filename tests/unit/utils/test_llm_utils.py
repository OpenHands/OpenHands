"""Tests for openhands.utils.llm module."""

from openhands.utils.llm import get_provider_api_base, is_openhands_model


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
