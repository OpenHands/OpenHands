"""Tests for the SlackPlugin entry point."""

from integrations.slack.plugin import SlackPlugin, SlackPluginConfig


class TestSlackPluginConfig:
    """Test SlackPluginConfig defaults and construction."""

    def test_default_config(self):
        config = SlackPluginConfig()
        assert config.client_id is None
        assert config.client_secret is None
        assert config.signing_secret is None
        assert config.webhooks_enabled is False
        assert config.keycloak_client_id == ''
        assert config.host_url == ''

    def test_custom_config(self):
        config = SlackPluginConfig(
            client_id='test-client-id',
            client_secret='test-secret',
            signing_secret='test-signing-secret',
            webhooks_enabled=True,
            host_url='https://example.com',
        )
        assert config.client_id == 'test-client-id'
        assert config.webhooks_enabled is True
        assert config.host_url == 'https://example.com'


class TestSlackPlugin:
    """Test SlackPlugin lifecycle and router creation."""

    def test_plugin_name(self):
        plugin = SlackPlugin(SlackPluginConfig())
        assert plugin.name == 'slack'

    def test_plugin_initialize_and_shutdown(self):
        plugin = SlackPlugin(SlackPluginConfig())
        assert plugin._initialized is False

        plugin.initialize()
        assert plugin._initialized is True

        # Double init is a no-op
        plugin.initialize()  # type: ignore[unreachable]
        assert plugin._initialized is True

        plugin.shutdown()
        assert plugin._initialized is False

    def test_plugin_config_accessible(self):
        config = SlackPluginConfig(client_id='test-id')
        plugin = SlackPlugin(config)
        assert plugin.config.client_id == 'test-id'

    def test_plugin_config_with_all_fields(self):
        config = SlackPluginConfig(
            client_id='cid',
            client_secret='csecret',
            signing_secret='ssecret',
            webhooks_enabled=True,
            keycloak_client_id='kc-cid',
            keycloak_realm_name='kc-realm',
            keycloak_server_url_ext='https://kc.example.com',
            host_url='https://app.example.com',
            jwt_secret='jwt-secret',
        )
        assert config.client_id == 'cid'
        assert config.client_secret == 'csecret'
        assert config.signing_secret == 'ssecret'
        assert config.webhooks_enabled is True
        assert config.keycloak_client_id == 'kc-cid'
        assert config.keycloak_realm_name == 'kc-realm'
        assert config.keycloak_server_url_ext == 'https://kc.example.com'
        assert config.host_url == 'https://app.example.com'
        assert config.jwt_secret == 'jwt-secret'
