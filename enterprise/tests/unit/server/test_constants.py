"""Tests for enterprise server constants, specifically DEPLOYMENT_MODE detection."""

from unittest.mock import patch

import pytest


class TestDeploymentMode:
    """Tests for _get_deployment_mode() and _is_all_hands_managed_domain() functions."""

    @pytest.fixture(autouse=True)
    def _no_explicit_mode(self, monkeypatch):
        """Host-heuristic tests must ignore any ambient OH_DEPLOYMENT_MODE."""
        monkeypatch.delenv('OH_DEPLOYMENT_MODE', raising=False)

    @pytest.mark.parametrize(
        'web_host,expected_mode',
        [
            # All-Hands managed domains should return 'cloud'
            ('app.all-hands.dev', 'cloud'),
            ('staging.all-hands.dev', 'cloud'),
            ('feature-123.staging.all-hands.dev', 'cloud'),
            ('pr-456.staging.all-hands.dev', 'cloud'),
            ('app.openhands.ai', 'cloud'),
            # openhands.dev is also an All-Hands managed domain
            ('app.openhands.dev', 'cloud'),
            ('staging.openhands.dev', 'cloud'),
            ('pr-456.staging.openhands.dev', 'cloud'),
            # Customer domains should return 'self_hosted'
            ('openhands.acme.com', 'self_hosted'),
            ('internal.company.io', 'self_hosted'),
            ('dev.mycompany.net', 'self_hosted'),
            ('openhands.example.org', 'self_hosted'),
            ('localhost', 'self_hosted'),  # localhost is not a managed domain
            # Edge cases
            ('all-hands.dev', 'self_hosted'),  # Not a subdomain, so not managed
            ('fake-all-hands.dev', 'self_hosted'),
            ('app.all-hands.dev.evil.com', 'self_hosted'),
        ],
    )
    def test_deployment_mode_detection(self, web_host: str, expected_mode: str):
        """Test that DEPLOYMENT_MODE is correctly determined based on WEB_HOST."""
        with patch.dict('os.environ', {'WEB_HOST': web_host}):
            # Need to reimport to pick up the mocked environment variable
            import importlib

            import server.constants as constants_module

            importlib.reload(constants_module)

            assert constants_module.DEPLOYMENT_MODE == expected_mode

    @pytest.mark.parametrize(
        'flag,web_host,expected_mode',
        [
            # Explicit flag wins over the host heuristic
            ('self_hosted', 'app.all-hands.dev', 'self_hosted'),
            ('cloud', 'openhands.acme.com', 'cloud'),
            # Case/whitespace tolerant
            ('  Self_Hosted ', 'app.all-hands.dev', 'self_hosted'),
            # Invalid/empty values fall back to the host heuristic
            ('bogus', 'app.all-hands.dev', 'cloud'),
            ('', 'openhands.acme.com', 'self_hosted'),
        ],
    )
    def test_explicit_deployment_mode_overrides_host(
        self, flag: str, web_host: str, expected_mode: str
    ):
        """OH_DEPLOYMENT_MODE takes precedence; invalid values fall back to WEB_HOST."""
        with patch.dict(
            'os.environ', {'OH_DEPLOYMENT_MODE': flag, 'WEB_HOST': web_host}
        ):
            import importlib

            import server.constants as constants_module

            importlib.reload(constants_module)

            assert constants_module.DEPLOYMENT_MODE == expected_mode

    @pytest.mark.parametrize(
        'host,expected',
        [
            ('app.all-hands.dev', True),
            ('staging.all-hands.dev', True),
            ('feature.staging.all-hands.dev', True),
            ('app.openhands.ai', True),
            ('app.openhands.dev', True),
            ('staging.openhands.dev', True),
            ('pr-1.staging.openhands.dev', True),
            ('localhost', False),  # localhost is not a managed domain
            ('customer.example.com', False),
            ('all-hands.dev', False),
            ('openhands.dev', False),  # apex is not a subdomain, so not managed
        ],
    )
    def test_is_all_hands_managed_domain(self, host: str, expected: bool):
        """Test _is_all_hands_managed_domain() helper function."""
        from server.constants import _is_all_hands_managed_domain

        assert _is_all_hands_managed_domain(host) == expected

    def test_deployment_mode_default_is_cloud(self):
        """Test that default WEB_HOST (app.all-hands.dev) results in 'cloud' mode."""
        with patch.dict('os.environ', {}, clear=True):
            # Remove WEB_HOST to test default
            import importlib
            import os

            if 'WEB_HOST' in os.environ:
                del os.environ['WEB_HOST']

            import server.constants as constants_module

            importlib.reload(constants_module)

            # Default WEB_HOST is 'app.all-hands.dev' which should be 'cloud'
            assert constants_module.DEPLOYMENT_MODE == 'cloud'


class TestStagingAndFeatureEnvDetection:
    """IS_STAGING_ENV / IS_FEATURE_ENV must recognize both the legacy
    all-hands.dev and the new openhands.dev staging/feature hosts."""

    @pytest.mark.parametrize(
        'web_host,is_staging,is_feature',
        [
            # Bare staging hosts: a staging env, but NOT a feature env
            ('staging.all-hands.dev', True, False),
            ('staging.openhands.dev', True, False),
            # Feature / preview hosts on both domains
            ('feature-123.staging.all-hands.dev', True, True),
            ('pr-279.staging.all-hands.dev', True, True),
            ('pr-279.staging.openhands.dev', True, True),
            ('feature-123.staging.openhands.dev', True, True),
            # Platform-team sandbox
            ('pr-279.ohe-staging.platform-team.all-hands.dev', True, True),
            # Production / customer / local hosts are neither
            ('app.all-hands.dev', False, False),
            ('app.openhands.dev', False, False),
            ('openhands.acme.com', False, False),
            ('localhost', False, False),
        ],
    )
    def test_staging_and_feature_env_detection(
        self, web_host: str, is_staging: bool, is_feature: bool
    ):
        """WEB_HOST drives IS_STAGING_ENV / IS_FEATURE_ENV for both domains."""
        with patch.dict('os.environ', {'WEB_HOST': web_host}):
            import importlib

            import server.constants as constants_module

            importlib.reload(constants_module)

            assert constants_module.IS_STAGING_ENV is is_staging
            assert constants_module.IS_FEATURE_ENV is is_feature


class TestDeploymentModeInConfig:
    """Tests for DEPLOYMENT_MODE being exposed in config API."""

    def test_deployment_mode_included_in_feature_flags(self):
        """Test that DEPLOYMENT_MODE is included in FEATURE_FLAGS from get_config()."""
        from server.config import SaaSServerConfig

        with patch('server.config.DEPLOYMENT_MODE', 'cloud'):
            saas_config = SaaSServerConfig()
            config = saas_config.get_config()

            assert 'FEATURE_FLAGS' in config
            assert 'DEPLOYMENT_MODE' in config['FEATURE_FLAGS']
            assert config['FEATURE_FLAGS']['DEPLOYMENT_MODE'] == 'cloud'

    def test_deployment_mode_self_hosted_in_feature_flags(self):
        """Test that self_hosted DEPLOYMENT_MODE is included in FEATURE_FLAGS."""
        from server.config import SaaSServerConfig

        with patch('server.config.DEPLOYMENT_MODE', 'self_hosted'):
            saas_config = SaaSServerConfig()
            config = saas_config.get_config()

            assert 'FEATURE_FLAGS' in config
            assert 'DEPLOYMENT_MODE' in config['FEATURE_FLAGS']
            assert config['FEATURE_FLAGS']['DEPLOYMENT_MODE'] == 'self_hosted'


class TestEnableAutomationsInConfig:
    """Tests for enable_automations flag in SaaSServerConfig and get_config()."""

    def test_enable_automations_true_in_feature_flags(self):
        """Test that ENABLE_AUTOMATIONS: True is included in FEATURE_FLAGS."""
        from server.config import SaaSServerConfig

        with patch('server.config.ENABLE_AUTOMATIONS', True):
            saas_config = SaaSServerConfig()
            saas_config.enable_automations = True
            config = saas_config.get_config()

            assert 'FEATURE_FLAGS' in config
            assert 'ENABLE_AUTOMATIONS' in config['FEATURE_FLAGS']
            assert config['FEATURE_FLAGS']['ENABLE_AUTOMATIONS'] is True

    def test_enable_automations_false_in_feature_flags(self):
        """Test that ENABLE_AUTOMATIONS: False is included in FEATURE_FLAGS."""
        from server.config import SaaSServerConfig

        with patch('server.config.ENABLE_AUTOMATIONS', False):
            saas_config = SaaSServerConfig()
            saas_config.enable_automations = False
            config = saas_config.get_config()

            assert 'FEATURE_FLAGS' in config
            assert 'ENABLE_AUTOMATIONS' in config['FEATURE_FLAGS']
            assert config['FEATURE_FLAGS']['ENABLE_AUTOMATIONS'] is False

    def test_enable_automations_defaults_to_true_for_saas(self):
        """Test that enable_automations defaults to True in SaaSServerConfig (SaaS default)."""
        import importlib

        import server.auth.constants as constants_module

        with patch.dict('os.environ', {}, clear=True):
            import os

            os.environ.pop('ENABLE_AUTOMATIONS', None)
            importlib.reload(constants_module)
            assert constants_module.ENABLE_AUTOMATIONS is True


class TestUserProvisioningEnabled:
    """Tests for the USER_PROVISIONING_ENABLED feature switch.

    The switch is driven by the ``USER_PROVISIONING_ENABLED`` env var
    (populated from the ``userProvisioning.enabled`` Helm value). It
    must accept both ``'true'`` and ``'1'`` because older Helm chart
    versions emit the latter form — accepting only one variant would
    silently disable the feature in those deployments. See AGENTS.md
    "Environment Variable Enable Toggles".
    """

    @pytest.mark.parametrize(
        'env_value,expected',
        [
            # Truthy variants accepted by the toggle convention.
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            # Falsy / unset / unknown values all disable the feature.
            ('false', False),
            ('False', False),
            ('0', False),
            ('', False),
            ('yes', False),  # Not part of the documented accepted set.
            ('on', False),
        ],
    )
    def test_user_provisioning_enabled_truthy_parsing(
        self, env_value: str, expected: bool
    ) -> None:
        with patch.dict('os.environ', {'USER_PROVISIONING_ENABLED': env_value}):
            import importlib

            import server.constants as constants_module

            importlib.reload(constants_module)
            assert constants_module.USER_PROVISIONING_ENABLED is expected

    def test_user_provisioning_enabled_default_is_false(self) -> None:
        """When the env var is unset, the feature must default to off."""
        with patch.dict('os.environ', {}, clear=True):
            import importlib
            import os

            os.environ.pop('USER_PROVISIONING_ENABLED', None)
            import server.constants as constants_module

            importlib.reload(constants_module)
            assert constants_module.USER_PROVISIONING_ENABLED is False


class TestOpenOrgCreationEnabled:
    """Tests for the OPEN_ORG_CREATION_ENABLED feature switch.

    Must accept both ``'true'`` and ``'1'`` per the documented enable-toggle
    convention so that older Helm chart versions emitting ``'1'`` keep the
    feature working. See AGENTS.md "Environment Variable Enable Toggles".
    """

    @pytest.mark.parametrize(
        'env_value,expected',
        [
            # Truthy variants accepted by the toggle convention.
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            # Falsy / unset / unknown values all disable the feature.
            ('false', False),
            ('False', False),
            ('0', False),
            ('', False),
            ('yes', False),  # Not part of the documented accepted set.
            ('on', False),
        ],
    )
    def test_open_org_creation_enabled_truthy_parsing(
        self, env_value: str, expected: bool
    ) -> None:
        with patch.dict('os.environ', {'OPEN_ORG_CREATION_ENABLED': env_value}):
            import importlib

            import server.constants as constants_module

            importlib.reload(constants_module)
            assert constants_module.OPEN_ORG_CREATION_ENABLED is expected

    def test_open_org_creation_enabled_default_is_false(self) -> None:
        """When the env var is unset, the feature must default to off."""
        with patch.dict('os.environ', {}, clear=True):
            import importlib
            import os

            os.environ.pop('OPEN_ORG_CREATION_ENABLED', None)
            import server.constants as constants_module

            importlib.reload(constants_module)
            assert constants_module.OPEN_ORG_CREATION_ENABLED is False
