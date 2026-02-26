"""Tests for JiraDcPlugin entry point."""

from unittest.mock import patch

from fastapi import APIRouter

from integrations.jira_dc.plugin import JiraDcPlugin, JiraDcPluginConfig


class TestJiraDcPluginConfig:
    """Test JiraDcPluginConfig defaults."""

    def test_defaults(self):
        cfg = JiraDcPluginConfig()
        assert cfg.webhooks_enabled is False
        assert cfg.base_url == ''
        assert cfg.client_id == ''
        assert cfg.client_secret == ''
        assert cfg.enable_oauth is True
        assert cfg.web_host == ''

    def test_custom_values(self):
        cfg = JiraDcPluginConfig(
            webhooks_enabled=True,
            base_url='https://jira.example.com',
            client_id='my-client-id',
            client_secret='my-secret',
            enable_oauth=False,
            web_host='app.example.com',
        )
        assert cfg.webhooks_enabled is True
        assert cfg.base_url == 'https://jira.example.com'
        assert cfg.client_id == 'my-client-id'
        assert cfg.client_secret == 'my-secret'
        assert cfg.enable_oauth is False
        assert cfg.web_host == 'app.example.com'


class TestJiraDcPlugin:
    """Test JiraDcPlugin lifecycle and router creation."""

    def test_init(self):
        cfg = JiraDcPluginConfig()
        plugin = JiraDcPlugin(cfg)
        assert plugin.config is cfg
        assert plugin._initialized is False

    def test_initialize_and_shutdown(self):
        cfg = JiraDcPluginConfig()
        plugin = JiraDcPlugin(cfg)
        assert plugin._initialized is False
        plugin.initialize()
        assert plugin._initialized is True
        plugin.shutdown()
        assert plugin._initialized is False

    @patch('integrations.jira_dc.routes.create_jira_dc_router')
    def test_get_router(self, mock_create_router):
        mock_router = APIRouter()
        mock_create_router.return_value = mock_router

        cfg = JiraDcPluginConfig()
        plugin = JiraDcPlugin(cfg)
        router = plugin.get_router()

        assert router is mock_router
        mock_create_router.assert_called_once_with(cfg)

    @patch('integrations.jira_dc.routes.create_jira_dc_router')
    def test_get_router_caches(self, mock_create_router):
        mock_router = APIRouter()
        mock_create_router.return_value = mock_router

        cfg = JiraDcPluginConfig()
        plugin = JiraDcPlugin(cfg)
        router1 = plugin.get_router()
        router2 = plugin.get_router()

        assert router1 is router2
        mock_create_router.assert_called_once()

    @patch('integrations.jira_dc.plugin.WEB_HOST', 'app.example.com')
    @patch('integrations.jira_dc.plugin.JIRA_DC_ENABLE_OAUTH', False)
    @patch('integrations.jira_dc.plugin.JIRA_DC_CLIENT_SECRET', 'secret')
    @patch('integrations.jira_dc.plugin.JIRA_DC_CLIENT_ID', 'client-id')
    @patch(
        'integrations.jira_dc.plugin.JIRA_DC_BASE_URL',
        'https://jira.example.com',
    )
    @patch.dict('os.environ', {'JIRA_DC_WEBHOOKS_ENABLED': '1'}, clear=False)
    def test_from_env(self):
        plugin = JiraDcPlugin.from_env()
        assert plugin.config.webhooks_enabled is True
        assert plugin.config.base_url == 'https://jira.example.com'
        assert plugin.config.client_id == 'client-id'
        assert plugin.config.client_secret == 'secret'
        assert plugin.config.enable_oauth is False
        assert plugin.config.web_host == 'app.example.com'
