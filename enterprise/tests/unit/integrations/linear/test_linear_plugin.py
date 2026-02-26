"""Tests for LinearPlugin entry point."""

from unittest.mock import MagicMock, patch

from fastapi import APIRouter

from integrations.linear.plugin import LinearPlugin, LinearPluginConfig


class TestLinearPluginConfig:
    """Test LinearPluginConfig defaults."""

    def test_defaults(self):
        cfg = LinearPluginConfig()
        assert cfg.webhooks_enabled is False
        assert cfg.client_id == ""
        assert cfg.client_secret == ""
        assert cfg.web_host == ""

    def test_custom_values(self):
        cfg = LinearPluginConfig(
            webhooks_enabled=True,
            client_id="my-client-id",
            client_secret="my-secret",
            web_host="app.example.com",
        )
        assert cfg.webhooks_enabled is True
        assert cfg.client_id == "my-client-id"
        assert cfg.client_secret == "my-secret"
        assert cfg.web_host == "app.example.com"


class TestLinearPlugin:
    """Test LinearPlugin lifecycle and router creation."""

    def test_init(self):
        cfg = LinearPluginConfig()
        plugin = LinearPlugin(cfg)
        assert plugin.config is cfg
        assert plugin._initialized is False

    def test_initialize_and_shutdown(self):
        cfg = LinearPluginConfig()
        plugin = LinearPlugin(cfg)
        assert plugin._initialized is False
        plugin.initialize()
        assert plugin._initialized is True
        plugin.shutdown()
        assert plugin._initialized is False

    def test_get_router(self):
        mock_router = APIRouter()
        mock_module = MagicMock()
        mock_module.create_linear_router = MagicMock(return_value=mock_router)

        cfg = LinearPluginConfig()
        plugin = LinearPlugin(cfg)

        with patch.dict("sys.modules", {"integrations.linear.routes": mock_module}):
            router = plugin.get_router()

        assert router is mock_router
        mock_module.create_linear_router.assert_called_once_with(cfg)

    def test_get_router_caches(self):
        mock_router = APIRouter()
        mock_module = MagicMock()
        mock_module.create_linear_router = MagicMock(return_value=mock_router)

        cfg = LinearPluginConfig()
        plugin = LinearPlugin(cfg)

        with patch.dict("sys.modules", {"integrations.linear.routes": mock_module}):
            router1 = plugin.get_router()
            router2 = plugin.get_router()

        assert router1 is router2
        mock_module.create_linear_router.assert_called_once()

    @patch("server.constants.WEB_HOST", "app.example.com")
    @patch("server.auth.constants.LINEAR_CLIENT_SECRET", "secret")
    @patch("server.auth.constants.LINEAR_CLIENT_ID", "client-id")
    @patch.dict("os.environ", {"LINEAR_WEBHOOKS_ENABLED": "1"}, clear=False)
    def test_from_env(self):
        plugin = LinearPlugin.from_env()
        assert plugin.config.webhooks_enabled is True
        assert plugin.config.client_id == "client-id"
        assert plugin.config.client_secret == "secret"
        assert plugin.config.web_host == "app.example.com"
