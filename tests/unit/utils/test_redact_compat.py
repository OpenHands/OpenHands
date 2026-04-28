"""Tests for openhands.utils._redact_compat redaction utilities.

These tests verify that MCP config secrets are properly redacted before logging.
"""

from openhands.utils._redact_compat import (
    redact_api_key_literals,
    redact_text_secrets,
    redact_url_params,
    sanitize_config,
)

# The redaction placeholder
REDACTED = '<redacted>'


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

        # This is what the code does before logging - just str()
        log_output = redact_text_secrets(str(mcp_tools_to_sync))

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

        log_output = redact_text_secrets(str(sse_servers))

        assert 'sk-oh-realSessionKey456' not in log_output
        assert REDACTED in log_output
        assert 'http://localhost:8000/mcp/sse' in log_output  # URL should be visible


class TestSanitizeConfig:
    """Tests for sanitize_config (dict-based redaction with URL param handling)."""

    def test_redacts_url_query_params_in_mcp_config(self):
        """Reproduce the exact Datadog leak: tavilyApiKey in URL query params."""
        config = {
            'mcpServers': {
                'tavily': {
                    'url': 'https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-realkey123',
                    'transport': 'http',
                }
            }
        }
        sanitized = sanitize_config(config)
        assert 'tvly-realkey123' not in str(sanitized)
        assert 'mcp.tavily.com' in str(sanitized)

    def test_redacts_header_api_keys_in_mcp_config(self):
        """Reproduce the Datadog leak: X-Session-API-Key in headers."""
        config = {
            'mcpServers': {
                'myserver': {
                    'url': 'https://example.com/mcp',
                    'headers': {
                        'X-Session-API-Key': 'sk-oh-realsessionkey456',
                    },
                }
            }
        }
        sanitized = sanitize_config(config)
        assert 'sk-oh-realsessionkey456' not in str(sanitized)

    def test_combined_url_and_header_secrets(self):
        """Full scenario matching the production Datadog log pattern."""
        config = {
            'mcpServers': {
                'tavily': {
                    'url': 'https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-realkey123',
                    'transport': 'http',
                },
                'internal': {
                    'url': 'https://internal.example.com/mcp',
                    'headers': {
                        'X-Session-API-Key': 'sk-oh-realsessionkey456',
                    },
                },
            }
        }
        sanitized = sanitize_config(config)
        output = str(sanitized)
        assert 'tvly-realkey123' not in output
        assert 'sk-oh-realsessionkey456' not in output
        assert 'tavily' in output
        assert 'internal' in output
