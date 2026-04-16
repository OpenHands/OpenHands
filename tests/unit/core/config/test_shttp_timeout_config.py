"""Test MCP remote server timeout configuration."""

from openhands.core.config.mcp_config import RemoteMCPServer


class TestRemoteMCPServer:
    """Test remote server configuration with timeout field."""

    def test_remote_config_with_timeout(self):
        """Test remote config accepts timeout parameter."""
        config = RemoteMCPServer(
            url='https://api.example.com/mcp', transport='http', timeout=90
        )
        assert config.timeout == 90
        assert config.url == 'https://api.example.com/mcp'

    def test_remote_config_with_auth_and_timeout(self):
        """Test remote config with both auth and timeout."""
        config = RemoteMCPServer(
            url='https://api.example.com/mcp',
            transport='http',
            auth='test-key-123',
            timeout=120,
        )
        assert config.timeout == 120
        assert config.auth == 'test-key-123'

    def test_remote_config_default_timeout(self):
        """Test remote config default timeout (None)."""
        config = RemoteMCPServer(url='https://api.example.com/mcp', transport='http')
        assert config.timeout is None

    def test_remote_config_positive_timeout(self):
        """Test timeout accepts positive values."""
        valid_timeouts = [1, 5, 30, 60, 120, 300, 600, 1800, 3600]

        for timeout in valid_timeouts:
            config = RemoteMCPServer(
                url='https://api.example.com/mcp', transport='http', timeout=timeout
            )
            assert config.timeout == timeout

    def test_model_dump_includes_timeout(self):
        """Test that model serialization includes timeout field."""
        config = RemoteMCPServer(
            url='https://api.example.com/mcp',
            transport='http',
            auth='test-key',
            timeout=90,
        )

        data = config.model_dump()
        assert data['url'] == 'https://api.example.com/mcp'
        assert data['auth'] == 'test-key'
        assert data['timeout'] == 90

    def test_transport_types(self):
        """Test that transport field accepts sse and http."""
        sse = RemoteMCPServer(url='https://api.example.com/sse', transport='sse')
        assert sse.transport == 'sse'

        http = RemoteMCPServer(url='https://api.example.com/mcp', transport='http')
        assert http.transport == 'http'
