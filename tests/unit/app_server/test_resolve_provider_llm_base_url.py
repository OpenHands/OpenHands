"""Tests for resolve_provider_llm_base_url in openhands.app_server.config."""

from openhands.app_server.config import (
    _SDK_DEFAULT_PROXY,
    resolve_provider_llm_base_url,
)

SDK_DEFAULT = _SDK_DEFAULT_PROXY  # 'https://llm-proxy.app.all-hands.dev/'
STAGING_URL = 'https://llm-proxy.staging.all-hands.dev/'
CUSTOM_URL = 'https://my-own-proxy.example.com/v1'


class TestOpenHandsPrefixWithProviderUrl:
    """openhands/ prefix + SDK default URL + provider URL set → returns provider URL."""

    def test_openhands_prefix_replaces_sdk_default(self):
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=SDK_DEFAULT,
            provider_base_url=STAGING_URL,
        )
        assert result == STAGING_URL

    def test_openhands_prefix_sdk_default_no_trailing_slash(self):
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=SDK_DEFAULT.rstrip('/'),
            provider_base_url=STAGING_URL,
        )
        assert result == STAGING_URL


class TestOpenHandsPrefixWithCustomUrl:
    """openhands/ prefix + custom URL (not SDK default) → returns custom URL."""

    def test_custom_url_preserved(self):
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=CUSTOM_URL,
            provider_base_url=STAGING_URL,
        )
        assert result == CUSTOM_URL

    def test_custom_url_preserved_no_provider(self):
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=CUSTOM_URL,
            provider_base_url=None,
        )
        assert result == CUSTOM_URL


class TestOpenHandsPrefixNoProviderUrl:
    """openhands/ prefix + SDK default URL + no provider URL → returns SDK default."""

    def test_sdk_default_returned_when_no_provider(self):
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=SDK_DEFAULT,
            provider_base_url='',  # empty string = falsy
        )
        assert result == SDK_DEFAULT

    def test_sdk_default_returned_when_provider_none_and_env_unset(self, monkeypatch):
        monkeypatch.delenv('OPENHANDS_PROVIDER_BASE_URL', raising=False)
        monkeypatch.delenv('LLM_BASE_URL', raising=False)
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=SDK_DEFAULT,
            # provider_base_url defaults to None → falls back to env lookup
        )
        assert result == SDK_DEFAULT


class TestNonMatchingModelPrefix:
    """Model without openhands/ or litellm_proxy/ prefix → returns base_url unchanged."""

    def test_plain_model_returns_base_url(self):
        result = resolve_provider_llm_base_url(
            model='gpt-4',
            base_url='https://api.openai.com/v1',
            provider_base_url=STAGING_URL,
        )
        assert result == 'https://api.openai.com/v1'

    def test_anthropic_model_returns_base_url(self):
        result = resolve_provider_llm_base_url(
            model='anthropic/claude-3-opus',
            base_url='https://api.anthropic.com',
            provider_base_url=STAGING_URL,
        )
        assert result == 'https://api.anthropic.com'

    def test_none_base_url_passthrough(self):
        result = resolve_provider_llm_base_url(
            model='gpt-4',
            base_url=None,
            provider_base_url=STAGING_URL,
        )
        assert result is None


class TestLitellmProxyPrefix:
    """litellm_proxy/ prefix behaves identically to openhands/ prefix."""

    def test_replaces_sdk_default(self):
        result = resolve_provider_llm_base_url(
            model='litellm_proxy/gpt-4',
            base_url=SDK_DEFAULT,
            provider_base_url=STAGING_URL,
        )
        assert result == STAGING_URL

    def test_custom_url_preserved(self):
        result = resolve_provider_llm_base_url(
            model='litellm_proxy/gpt-4',
            base_url=CUSTOM_URL,
            provider_base_url=STAGING_URL,
        )
        assert result == CUSTOM_URL

    def test_sdk_default_returned_when_no_provider(self):
        result = resolve_provider_llm_base_url(
            model='litellm_proxy/gpt-4',
            base_url=SDK_DEFAULT,
            provider_base_url='',
        )
        assert result == SDK_DEFAULT


class TestEdgeCases:
    """Edge cases: None values, empty strings, trailing slash normalization."""

    def test_none_model_returns_base_url(self):
        result = resolve_provider_llm_base_url(
            model=None,
            base_url=SDK_DEFAULT,
            provider_base_url=STAGING_URL,
        )
        assert result == SDK_DEFAULT

    def test_empty_model_returns_base_url(self):
        result = resolve_provider_llm_base_url(
            model='',
            base_url=SDK_DEFAULT,
            provider_base_url=STAGING_URL,
        )
        assert result == SDK_DEFAULT

    def test_none_base_url_with_openhands_model_and_provider(self):
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=None,
            provider_base_url=STAGING_URL,
        )
        assert result == STAGING_URL

    def test_none_base_url_with_openhands_model_no_provider(self, monkeypatch):
        monkeypatch.delenv('OPENHANDS_PROVIDER_BASE_URL', raising=False)
        monkeypatch.delenv('LLM_BASE_URL', raising=False)
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=None,
        )
        assert result is None

    def test_trailing_slash_normalization(self):
        """SDK default with and without trailing slash should both be detected."""
        for url in [SDK_DEFAULT, SDK_DEFAULT.rstrip('/')]:
            result = resolve_provider_llm_base_url(
                model='openhands/gpt-4',
                base_url=url,
                provider_base_url=STAGING_URL,
            )
            assert result == STAGING_URL, f'Failed for base_url={url!r}'

    def test_env_fallback_when_provider_base_url_is_none(self, monkeypatch):
        monkeypatch.setenv('OPENHANDS_PROVIDER_BASE_URL', STAGING_URL)
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=SDK_DEFAULT,
            # provider_base_url defaults to None → env lookup
        )
        assert result == STAGING_URL

    def test_llm_base_url_env_fallback(self, monkeypatch):
        monkeypatch.delenv('OPENHANDS_PROVIDER_BASE_URL', raising=False)
        monkeypatch.setenv('LLM_BASE_URL', STAGING_URL)
        result = resolve_provider_llm_base_url(
            model='openhands/gpt-4',
            base_url=SDK_DEFAULT,
        )
        assert result == STAGING_URL
