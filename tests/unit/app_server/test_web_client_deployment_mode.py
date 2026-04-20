"""Tests for web_client_deployment_mode module."""

from unittest.mock import patch

import pytest

from openhands.app_server.web_client.web_client_deployment_mode import (
    get_deployment_mode,
)


class TestGetDeploymentMode:
    """Tests for get_deployment_mode() function."""

    @pytest.mark.parametrize(
        'web_host,expected',
        [
            # All-Hands managed domains should return 'cloud'
            ('app.all-hands.dev', 'cloud'),
            ('staging.all-hands.dev', 'cloud'),
            ('feature-123.staging.all-hands.dev', 'cloud'),
            ('app.openhands.ai', 'cloud'),
            ('subdomain.openhands.ai', 'cloud'),
            # Customer domains should return 'self_hosted'
            ('openhands.acme.com', 'self_hosted'),
            ('internal.company.io', 'self_hosted'),
            ('dev.mycompany.net', 'self_hosted'),
            # Edge cases - not subdomains
            ('all-hands.dev', 'self_hosted'),
            ('openhands.ai', 'self_hosted'),
            # Malicious domains
            ('fake-all-hands.dev', 'self_hosted'),
            ('app.all-hands.dev.evil.com', 'self_hosted'),
        ],
    )
    def test_deployment_mode_detection(self, web_host: str, expected: str):
        """Test that deployment mode is correctly determined based on WEB_HOST."""
        with patch.dict('os.environ', {'WEB_HOST': web_host}, clear=False):
            result = get_deployment_mode()
            assert result == expected

    def test_returns_none_when_web_host_empty(self):
        """Test that empty WEB_HOST returns None."""
        with patch.dict('os.environ', {'WEB_HOST': ''}, clear=False):
            result = get_deployment_mode()
            assert result is None

    def test_returns_none_when_web_host_not_set(self):
        """Test that missing WEB_HOST returns None."""
        with patch.dict('os.environ', {}, clear=True):
            result = get_deployment_mode()
            assert result is None

    def test_returns_none_when_web_host_whitespace_only(self):
        """Test that whitespace-only WEB_HOST returns None."""
        with patch.dict('os.environ', {'WEB_HOST': '   '}, clear=False):
            result = get_deployment_mode()
            assert result is None

    def test_oh_web_host_takes_precedence_over_web_host(self):
        """Test that OH_WEB_HOST takes precedence over WEB_HOST."""
        with patch.dict(
            'os.environ',
            {'OH_WEB_HOST': 'app.all-hands.dev', 'WEB_HOST': 'customer.example.com'},
            clear=False,
        ):
            result = get_deployment_mode()
            assert result == 'cloud'

    def test_falls_back_to_web_host_when_oh_web_host_not_set(self):
        """Test that WEB_HOST is used when OH_WEB_HOST is not set."""
        with patch.dict(
            'os.environ', {'WEB_HOST': 'customer.example.com'}, clear=False
        ):
            # Ensure OH_WEB_HOST is not set
            import os

            if 'OH_WEB_HOST' in os.environ:
                del os.environ['OH_WEB_HOST']
            result = get_deployment_mode()
            assert result == 'self_hosted'
