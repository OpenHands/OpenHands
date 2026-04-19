"""Tests for enterprise server constants, specifically DEPLOYMENT_MODE detection."""

from unittest.mock import patch

import pytest


class TestDeploymentMode:
    """Tests for _get_deployment_mode() and _is_all_hands_managed_domain() functions."""

    @pytest.mark.parametrize(
        'web_host,expected_mode',
        [
            # All-Hands managed domains should return 'cloud'
            ('app.all-hands.dev', 'cloud'),
            ('staging.all-hands.dev', 'cloud'),
            ('feature-123.staging.all-hands.dev', 'cloud'),
            ('pr-456.staging.all-hands.dev', 'cloud'),
            ('app.openhands.ai', 'cloud'),
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
        'host,expected',
        [
            ('app.all-hands.dev', True),
            ('staging.all-hands.dev', True),
            ('feature.staging.all-hands.dev', True),
            ('app.openhands.ai', True),
            ('localhost', False),  # localhost is not a managed domain
            ('customer.example.com', False),
            ('all-hands.dev', False),
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
