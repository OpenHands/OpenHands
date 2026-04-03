"""Tests for openhands.utils.log_utils credential redaction."""

from openhands.utils.log_utils import redact_mcp_config, redact_mcp_config_model


class TestRedactMcpConfig:
    """Tests for the V1 dict-format MCP config redaction."""

    def test_redacts_session_api_key_header(self):
        config = {
            "mcpServers": {
                "default": {
                    "url": "https://example.com/mcp/mcp",
                    "headers": {
                        "X-OpenHands-ServerConversation-ID": "conv-123",
                        "X-Session-API-Key": "sk-oh-secret-key-12345",
                    },
                }
            }
        }
        result = redact_mcp_config(config)
        assert result["mcpServers"]["default"]["headers"]["X-Session-API-Key"] == "***"
        # Non-sensitive header should be preserved
        assert (
            result["mcpServers"]["default"]["headers"][
                "X-OpenHands-ServerConversation-ID"
            ]
            == "conv-123"
        )

    def test_redacts_tavily_api_key_in_url(self):
        config = {
            "mcpServers": {
                "tavily": {
                    "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-prod-secret123"
                }
            }
        }
        result = redact_mcp_config(config)
        assert "tvly-prod-secret123" not in result["mcpServers"]["tavily"]["url"]
        assert "tavilyApiKey=***" in result["mcpServers"]["tavily"]["url"]

    def test_redacts_authorization_header(self):
        config = {
            "mcpServers": {
                "custom_sse": {
                    "url": "https://custom-mcp.example.com/sse",
                    "transport": "sse",
                    "headers": {"Authorization": "Bearer my-secret-token"},
                }
            }
        }
        result = redact_mcp_config(config)
        assert result["mcpServers"]["custom_sse"]["headers"]["Authorization"] == "***"

    def test_redacts_env_vars(self):
        config = {
            "mcpServers": {
                "tavily_stdio": {
                    "command": "npx",
                    "args": ["-y", "tavily-mcp@0.2.1"],
                    "env": {
                        "TAVILY_API_KEY": "tvly-prod-secret123",
                        "HOME": "/home/user",
                    },
                }
            }
        }
        result = redact_mcp_config(config)
        assert result["mcpServers"]["tavily_stdio"]["env"]["TAVILY_API_KEY"] == "***"
        assert result["mcpServers"]["tavily_stdio"]["env"]["HOME"] == "/home/user"

    def test_does_not_mutate_original(self):
        config = {
            "mcpServers": {
                "default": {
                    "url": "https://example.com/mcp",
                    "headers": {"X-Session-API-Key": "sk-oh-secret"},
                }
            }
        }
        redact_mcp_config(config)
        # Original should be unchanged
        assert (
            config["mcpServers"]["default"]["headers"]["X-Session-API-Key"]
            == "sk-oh-secret"
        )

    def test_handles_empty_config(self):
        assert redact_mcp_config({}) == {}
        assert redact_mcp_config({"mcpServers": {}}) == {"mcpServers": {}}

    def test_preserves_non_sensitive_fields(self):
        config = {
            "mcpServers": {
                "default": {
                    "url": "https://example.com/mcp/mcp",
                    "transport": "streamable-http",
                    "timeout": 30,
                }
            }
        }
        result = redact_mcp_config(config)
        assert result == config

    def test_redacts_multiple_servers(self):
        config = {
            "mcpServers": {
                "default": {
                    "url": "https://example.com/mcp",
                    "headers": {"X-Session-API-Key": "sk-oh-secret"},
                },
                "tavily": {
                    "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-prod-key"
                },
                "custom": {
                    "url": "https://custom.example.com/mcp",
                    "headers": {"Authorization": "Bearer token123"},
                },
            }
        }
        result = redact_mcp_config(config)
        assert result["mcpServers"]["default"]["headers"]["X-Session-API-Key"] == "***"
        assert "tvly-prod-key" not in result["mcpServers"]["tavily"]["url"]
        assert result["mcpServers"]["custom"]["headers"]["Authorization"] == "***"

    def test_redacts_api_key_field(self):
        config = {
            "mcpServers": {
                "server": {
                    "url": "https://example.com",
                    "api_key": "secret-key-value",
                }
            }
        }
        result = redact_mcp_config(config)
        assert result["mcpServers"]["server"]["api_key"] == "***"


class TestRedactMcpConfigModel:
    """Tests for the V0 pydantic-model string redaction."""

    def test_redacts_api_key_in_string(self):
        text = "MCPConfig(sse_servers=[MCPSSEServerConfig(url='https://example.com', api_key='secret123')])"
        result = redact_mcp_config_model(text)
        assert "secret123" not in result
        assert "api_key='***'" in result

    def test_redacts_tavily_api_key_in_url(self):
        text = "url='https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-prod-secret123'"
        result = redact_mcp_config_model(text)
        assert "tvly-prod-secret123" not in result
        assert "tavilyApiKey=***" in result

    def test_redacts_env_var_in_string(self):
        text = "env={'TAVILY_API_KEY': 'tvly-prod-secret123', 'HOME': '/home/user'}"
        result = redact_mcp_config_model(text)
        assert "tvly-prod-secret123" not in result
        assert "/home/user" in result

    def test_redacts_authorization_header(self):
        text = "headers={'Authorization': 'Bearer my-secret-token'}"
        result = redact_mcp_config_model(text)
        assert "my-secret-token" not in result

    def test_redacts_session_api_key_header(self):
        text = "headers={'X-Session-API-Key': 'sk-oh-secret-key'}"
        result = redact_mcp_config_model(text)
        assert "sk-oh-secret-key" not in result

    def test_preserves_non_sensitive_data(self):
        text = "MCPConfig(sse_servers=[], shttp_servers=[MCPSHTTPServerConfig(url='https://example.com')], stdio_servers=[])"
        result = redact_mcp_config_model(text)
        assert "https://example.com" in result
