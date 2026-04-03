"""Tests for app_server config helper functions."""

import os
from unittest.mock import patch


class TestGetDefaultWebUrl:
    """Test cases for get_default_web_url."""

    def test_returns_none_when_web_host_unset(self):
        from openhands.app_server.config import get_default_web_url

        with patch.dict(os.environ, {}, clear=True):
            assert get_default_web_url() is None

    def test_returns_https_url_when_web_host_set(self):
        from openhands.app_server.config import get_default_web_url

        with patch.dict(os.environ, {'WEB_HOST': 'example.com'}):
            assert get_default_web_url() == 'https://example.com'

    def test_strips_whitespace_from_web_host(self):
        from openhands.app_server.config import get_default_web_url

        with patch.dict(os.environ, {'WEB_HOST': '  example.com  '}):
            assert get_default_web_url() == 'https://example.com'


class TestGetOpenHandsProviderBaseUrl:
    """Test cases for get_openhands_provider_base_url."""

    def test_returns_none_when_provider_base_url_unset(self):
        from openhands.app_server.config import get_openhands_provider_base_url

        with patch.dict(os.environ, {}, clear=True):
            assert get_openhands_provider_base_url() is None

    def test_returns_provider_base_url_when_set(self):
        from openhands.app_server.config import get_openhands_provider_base_url

        with patch.dict(
            os.environ,
            {'OPENHANDS_PROVIDER_BASE_URL': 'https://provider.example.com'},
        ):
            assert (
                get_openhands_provider_base_url()
                == 'https://provider.example.com'
            )

    def test_strips_whitespace_from_provider_base_url(self):
        from openhands.app_server.config import get_openhands_provider_base_url

        with patch.dict(
            os.environ,
            {'OPENHANDS_PROVIDER_BASE_URL': '  https://provider.example.com  '},
        ):
            assert (
                get_openhands_provider_base_url()
                == 'https://provider.example.com'
            )
