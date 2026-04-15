"""Tests for openhands.utils._redact_compat redaction utilities.

These tests verify that MCP config secrets are properly redacted before logging.
"""

import json

from openhands.utils._redact_compat import (
    redact_api_key_literals,
    redact_text_secrets,
    redact_url_params,
    sanitize_config,
    sanitize_for_logging,
)

# The SDK uses '<redacted>' as the placeholder
REDACTED = '<redacted>'


class TestSanitizeForLogging:
    """Tests for sanitize_for_logging which handles both dicts and lists."""

    def test_sanitize_list_of_mcp_stdio_configs(self):
        """Test that MCP stdio server configs with env secrets are redacted."""
        mcp_tools = [
            {
                'name': 'tavily',
                'command': 'npx',
                'args': ['-y', '@tavily/mcp-server'],
                'env': {'TAVILY_API_KEY': 'tvly-abc123secretkey'},
            },
            {
                'name': 'other-tool',
                'command': 'other-cmd',
                'env': {'OTHER_KEY': 'other-value', 'API_TOKEN': 'secret-token-123'},
            },
        ]

        sanitized = sanitize_for_logging(mcp_tools)

        # TAVILY_API_KEY should be redacted
        assert sanitized[0]['env']['TAVILY_API_KEY'] == REDACTED
        # API_TOKEN should be redacted (contains TOKEN)
        assert sanitized[1]['env']['API_TOKEN'] == REDACTED
        # Non-sensitive values should remain
        assert sanitized[0]['name'] == 'tavily'
        assert sanitized[0]['command'] == 'npx'

    def test_sanitize_list_of_mcp_sse_configs(self):
        """Test that MCP SSE server configs with api_key are redacted."""
        sse_servers = [
            {
                'url': 'http://localhost:8000/mcp/sse',
                'api_key': 'sk-oh-abc123sessionkey',
            },
            {
                'url': 'http://other-server/sse',
                'api_key': 'another-api-key-value',
            },
        ]

        sanitized = sanitize_for_logging(sse_servers)

        # api_key should be redacted
        assert sanitized[0]['api_key'] == REDACTED
        assert sanitized[1]['api_key'] == REDACTED
        # URLs should remain
        assert sanitized[0]['url'] == 'http://localhost:8000/mcp/sse'

    def test_sanitize_dict_with_secrets(self):
        """Test that dicts with secret keys are redacted."""
        config = {
            'name': 'test',
            'password': 'secret123',
            'api_key': 'my-api-key',
            'OPENAI_API_KEY': 'sk-proj-abc123',
        }

        sanitized = sanitize_for_logging(config)

        assert sanitized['password'] == REDACTED
        assert sanitized['api_key'] == REDACTED
        assert sanitized['OPENAI_API_KEY'] == REDACTED
        assert sanitized['name'] == 'test'

    def test_sanitize_does_not_mutate_original(self):
        """Test that sanitization creates a deep copy."""
        original = [{'name': 'test', 'env': {'API_KEY': 'secret'}}]

        sanitize_for_logging(original)

        # Original should not be modified
        assert original[0]['env']['API_KEY'] == 'secret'

    def test_sanitize_empty_list(self):
        """Test sanitizing an empty list."""
        assert sanitize_for_logging([]) == []

    def test_sanitize_nested_secrets(self):
        """Test that nested secrets in env dicts are redacted.

        Note: The SDK's sanitize_dict redacts ALL values inside 'env' dicts
        since environment variables typically contain sensitive data.
        """
        config = [
            {
                'name': 'nested',
                'env': {
                    'NESTED_SECRET_KEY': 'should-be-hidden',
                    'NORMAL_VAR': 'also-hidden-in-env',
                },
            }
        ]

        sanitized = sanitize_for_logging(config)

        # ALL env values are redacted (SDK's security-first approach)
        assert sanitized[0]['env']['NESTED_SECRET_KEY'] == REDACTED
        assert sanitized[0]['env']['NORMAL_VAR'] == REDACTED
        # Non-env fields remain visible
        assert sanitized[0]['name'] == 'nested'

    def test_sanitize_url_with_api_key_param(self):
        """Test that URLs with sensitive query params are redacted."""
        config = [
            {
                'url': 'https://api.example.com?apiKey=secret123&other=visible',
                'name': 'test',
            }
        ]

        sanitized = sanitize_for_logging(config)

        assert 'secret123' not in sanitized[0]['url']
        assert 'other=visible' in sanitized[0]['url']

    def test_sanitize_x_session_api_key_header(self):
        """Test that X-Session-API-Key headers are redacted.

        Note: The SDK's sanitize_dict redacts ALL values inside 'headers' dicts
        since headers often contain sensitive auth data.
        """
        config = {
            'headers': {
                'X-Session-API-Key': 'sk-oh-secret123',
                'Content-Type': 'application/json',
            }
        }

        sanitized = sanitize_for_logging(config)

        # ALL header values are redacted (SDK's security-first approach)
        assert sanitized['headers']['X-Session-API-Key'] == REDACTED
        assert sanitized['headers']['Content-Type'] == REDACTED


class TestSanitizeConfig:
    """Tests for sanitize_config (dict-only version)."""

    def test_sanitize_mcp_config_dict(self):
        """Test sanitizing a full MCP config dict."""
        config = {
            'sse_servers': [
                {'url': 'http://localhost/sse', 'api_key': 'secret-key'},
            ],
            'stdio_servers': [
                {'name': 'tavily', 'env': {'TAVILY_API_KEY': 'tvly-secret'}},
            ],
        }

        sanitized = sanitize_config(config)

        assert sanitized['sse_servers'][0]['api_key'] == REDACTED
        assert sanitized['stdio_servers'][0]['env']['TAVILY_API_KEY'] == REDACTED


class TestRedactTextSecrets:
    """Tests for redact_text_secrets (string-based redaction)."""

    def test_redact_api_key_in_string_repr(self):
        """Test redacting api_key='...' patterns."""
        text = "MCPSSEServerConfig(url='http://localhost', api_key='secret123')"
        redacted = redact_text_secrets(text)
        assert "api_key='<redacted>'" in redacted
        assert 'secret123' not in redacted

    def test_redact_env_dict_in_string(self):
        """Test redacting env dict secrets in string representation."""
        text = "{'TAVILY_API_KEY': 'tvly-abc123', 'OTHER': 'visible'}"
        redacted = redact_text_secrets(text)
        assert 'tvly-abc123' not in redacted
        assert "'TAVILY_API_KEY': '<redacted>'" in redacted

    def test_redact_x_session_api_key_header(self):
        """Test redacting X-Session-API-Key header in string."""
        text = "{'X-Session-API-Key': 'sk-oh-sessionkey123'}"
        redacted = redact_text_secrets(text)
        assert 'sk-oh-sessionkey123' not in redacted


class TestRedactApiKeyLiterals:
    """Tests for redact_api_key_literals (pattern-based token redaction)."""

    def test_redact_tavily_key(self):
        """Test that Tavily API keys are redacted."""
        text = 'Using key tvly-abc123secretkey for search'
        redacted = redact_api_key_literals(text)
        assert 'tvly-abc123secretkey' not in redacted
        assert '<redacted>' in redacted

    def test_redact_openai_key(self):
        """Test that OpenAI API keys are redacted.

        Note: The regex requires at least 20 chars after the prefix.
        """
        text = 'API key is sk-proj-abc123xyz456def789ghi012'
        redacted = redact_api_key_literals(text)
        assert 'sk-proj-abc123xyz456def789ghi012' not in redacted

    def test_redact_openhands_session_token(self):
        """Test that OpenHands session tokens are redacted."""
        text = 'Session: sk-oh-abc123sessiontoken456'
        redacted = redact_api_key_literals(text)
        assert 'sk-oh-abc123sessiontoken456' not in redacted


class TestRedactUrlParams:
    """Tests for redact_url_params."""

    def test_redact_apikey_param(self):
        """Test redacting apiKey query parameter."""
        url = 'https://api.example.com/search?apiKey=secret123&query=test'
        redacted = redact_url_params(url)
        assert 'secret123' not in redacted
        # URL-encoded <redacted> is %3Credacted%3E
        assert 'apiKey=' in redacted
        assert 'query=test' in redacted

    def test_redact_token_param(self):
        """Test redacting token query parameter."""
        url = 'https://api.example.com?token=mytoken123'
        redacted = redact_url_params(url)
        assert 'mytoken123' not in redacted
        assert 'token=' in redacted


class TestMCPConfigLoggingIntegration:
    """Integration tests simulating actual MCP config logging scenarios."""

    def test_mcp_stdio_server_logging_is_safe(self):
        """Simulate logging MCP stdio server configs as done in action_execution_server.py."""
        mcp_tools_to_sync = [
            {
                'name': 'tavily',
                'command': 'npx',
                'args': ['-y', '@tavily/mcp-server'],
                'env': {'TAVILY_API_KEY': 'tvly-realSecretKey123'},
            }
        ]

        # This is what the code does before logging
        sanitized = sanitize_for_logging(mcp_tools_to_sync)
        log_output = json.dumps(sanitized, indent=2)

        assert 'tvly-realSecretKey123' not in log_output
        assert REDACTED in log_output
        assert 'tavily' in log_output  # Name should still be visible

    def test_mcp_sse_server_logging_is_safe(self):
        """Simulate logging MCP SSE server configs as done in action_execution_client.py."""
        sse_servers = [
            {
                'url': 'http://localhost:8000/mcp/sse',
                'api_key': 'sk-oh-realSessionKey456',
            }
        ]

        sanitized = sanitize_for_logging(sse_servers)
        log_output = str(sanitized)

        assert 'sk-oh-realSessionKey456' not in log_output
        assert REDACTED in log_output
        assert 'http://localhost:8000/mcp/sse' in log_output  # URL should be visible
