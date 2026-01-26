"""Tests for openhands.utils.http_session module."""

import importlib
import os
import ssl
from unittest import mock

import pytest


class TestGetVerifyCertificatesFromEnv:
    """Tests for _get_verify_certificates_from_env function."""

    def _reload_module(self):
        """Reload the http_session module to pick up new environment variables."""
        import openhands.utils.http_session as http_session_module

        importlib.reload(http_session_module)
        return http_session_module

    def test_default_returns_true(self):
        """By default, SSL verification should be enabled."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove any existing SSL-related env vars
            env = {k: v for k, v in os.environ.items() if k not in ('SSL_VERIFY', 'INSECURE_SKIP_VERIFY')}
            with mock.patch.dict(os.environ, env, clear=True):
                module = self._reload_module()
                assert module._verify_certificates is True

    def test_ssl_verify_false_disables_verification(self):
        """SSL_VERIFY=false should disable SSL verification."""
        with mock.patch.dict(os.environ, {'SSL_VERIFY': 'false'}, clear=True):
            module = self._reload_module()
            assert module._verify_certificates is False

    def test_ssl_verify_zero_disables_verification(self):
        """SSL_VERIFY=0 should disable SSL verification."""
        with mock.patch.dict(os.environ, {'SSL_VERIFY': '0'}, clear=True):
            module = self._reload_module()
            assert module._verify_certificates is False

    def test_ssl_verify_no_disables_verification(self):
        """SSL_VERIFY=no should disable SSL verification."""
        with mock.patch.dict(os.environ, {'SSL_VERIFY': 'no'}, clear=True):
            module = self._reload_module()
            assert module._verify_certificates is False

    def test_insecure_skip_verify_true_disables_verification(self):
        """INSECURE_SKIP_VERIFY=true should disable SSL verification."""
        with mock.patch.dict(os.environ, {'INSECURE_SKIP_VERIFY': 'true'}, clear=True):
            module = self._reload_module()
            assert module._verify_certificates is False

    def test_insecure_skip_verify_one_disables_verification(self):
        """INSECURE_SKIP_VERIFY=1 should disable SSL verification."""
        with mock.patch.dict(os.environ, {'INSECURE_SKIP_VERIFY': '1'}, clear=True):
            module = self._reload_module()
            assert module._verify_certificates is False

    def test_insecure_skip_verify_yes_disables_verification(self):
        """INSECURE_SKIP_VERIFY=yes should disable SSL verification."""
        with mock.patch.dict(os.environ, {'INSECURE_SKIP_VERIFY': 'yes'}, clear=True):
            module = self._reload_module()
            assert module._verify_certificates is False

    def test_ssl_verify_true_enables_verification(self):
        """SSL_VERIFY=true should keep SSL verification enabled."""
        with mock.patch.dict(os.environ, {'SSL_VERIFY': 'true'}, clear=True):
            module = self._reload_module()
            assert module._verify_certificates is True

    def test_ssl_verify_takes_precedence_over_insecure_skip(self):
        """SSL_VERIFY=false should take precedence over INSECURE_SKIP_VERIFY=false."""
        with mock.patch.dict(
            os.environ,
            {'SSL_VERIFY': 'false', 'INSECURE_SKIP_VERIFY': 'false'},
            clear=True,
        ):
            module = self._reload_module()
            assert module._verify_certificates is False

    def test_case_insensitive(self):
        """Environment variable values should be case-insensitive."""
        with mock.patch.dict(os.environ, {'SSL_VERIFY': 'FALSE'}, clear=True):
            module = self._reload_module()
            assert module._verify_certificates is False

        with mock.patch.dict(os.environ, {'INSECURE_SKIP_VERIFY': 'TRUE'}, clear=True):
            module = self._reload_module()
            assert module._verify_certificates is False


class TestHttpxVerifyOption:
    """Tests for httpx_verify_option function."""

    def _reload_module(self):
        """Reload the http_session module to pick up new environment variables."""
        import openhands.utils.http_session as http_session_module

        importlib.reload(http_session_module)
        return http_session_module

    def test_returns_ssl_context_when_verify_enabled(self):
        """Should return SSLContext when verification is enabled."""
        with mock.patch.dict(os.environ, {'SSL_VERIFY': 'true'}, clear=True):
            module = self._reload_module()
            result = module.httpx_verify_option()
            assert isinstance(result, ssl.SSLContext)

    def test_returns_false_when_verify_disabled(self):
        """Should return False when verification is disabled."""
        with mock.patch.dict(os.environ, {'SSL_VERIFY': 'false'}, clear=True):
            module = self._reload_module()
            result = module.httpx_verify_option()
            assert result is False
