"""Tests for URL utility functions that prevent URL hijacking attacks."""

from unittest.mock import MagicMock, patch

import pytest


class TestGetWebUrl:
    """Tests for get_web_url function."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request object."""
        request = MagicMock()
        request.url = MagicMock()
        return request

    def test_configured_web_url_is_used(self, mock_request):
        """When web_url is configured, it should be used instead of request URL."""
        from server.utils.url_utils import get_web_url

        mock_request.url.hostname = 'evil-attacker.com'
        mock_request.url.netloc = 'evil-attacker.com:443'

        mock_config = MagicMock()
        mock_config.web_url = 'https://app.all-hands.dev'

        with patch(
            'server.utils.url_utils.get_global_config', return_value=mock_config
        ):
            result = get_web_url(mock_request)

        assert result == 'https://app.all-hands.dev'
        # Should not use any info from the potentially poisoned request
        assert 'evil-attacker.com' not in result

    def test_configured_web_url_trailing_slash_stripped(self, mock_request):
        """Configured web_url should have trailing slashes stripped."""
        from server.utils.url_utils import get_web_url

        mock_config = MagicMock()
        mock_config.web_url = 'https://app.all-hands.dev/'

        with patch(
            'server.utils.url_utils.get_global_config', return_value=mock_config
        ):
            result = get_web_url(mock_request)

        assert result == 'https://app.all-hands.dev'
        assert not result.endswith('/')

    def test_unconfigured_web_url_localhost_uses_http(self, mock_request):
        """When web_url is not configured and hostname is localhost, use http."""
        from server.utils.url_utils import get_web_url

        mock_request.url.hostname = 'localhost'
        mock_request.url.netloc = 'localhost:3000'

        mock_config = MagicMock()
        mock_config.web_url = None

        with patch(
            'server.utils.url_utils.get_global_config', return_value=mock_config
        ):
            result = get_web_url(mock_request)

        assert result == 'http://localhost:3000'

    def test_unconfigured_web_url_non_localhost_uses_https(self, mock_request):
        """When web_url is not configured and hostname is not localhost, use https."""
        from server.utils.url_utils import get_web_url

        mock_request.url.hostname = 'example.com'
        mock_request.url.netloc = 'example.com:443'

        mock_config = MagicMock()
        mock_config.web_url = None

        with patch(
            'server.utils.url_utils.get_global_config', return_value=mock_config
        ):
            result = get_web_url(mock_request)

        assert result == 'https://example.com:443'

    def test_unconfigured_web_url_empty_string_fallback(self, mock_request):
        """Empty string web_url should trigger fallback."""
        from server.utils.url_utils import get_web_url

        mock_request.url.hostname = 'localhost'
        mock_request.url.netloc = 'localhost:3000'

        mock_config = MagicMock()
        mock_config.web_url = ''

        with patch(
            'server.utils.url_utils.get_global_config', return_value=mock_config
        ):
            result = get_web_url(mock_request)

        assert result == 'http://localhost:3000'


class TestGetCookieDomain:
    """Tests for get_cookie_domain function."""

    def test_production_with_configured_web_url(self):
        """In production with web_url configured, should return hostname."""
        from server.utils.url_utils import get_cookie_domain

        mock_config = MagicMock()
        mock_config.web_url = 'https://app.all-hands.dev'

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', False),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            result = get_cookie_domain()

        assert result == 'app.all-hands.dev'

    def test_production_without_web_url_returns_none(self):
        """In production without web_url configured, should return None."""
        from server.utils.url_utils import get_cookie_domain

        mock_config = MagicMock()
        mock_config.web_url = None

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', False),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            result = get_cookie_domain()

        assert result is None

    def test_local_env_returns_none(self):
        """In local environment, should return None for cookie domain."""
        from server.utils.url_utils import get_cookie_domain

        mock_config = MagicMock()
        mock_config.web_url = 'https://app.all-hands.dev'

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', False),
            patch('server.utils.url_utils.IS_LOCAL_ENV', True),
        ):
            result = get_cookie_domain()

        assert result is None

    def test_staging_env_returns_none(self):
        """In staging environment, should return None for cookie domain."""
        from server.utils.url_utils import get_cookie_domain

        mock_config = MagicMock()
        mock_config.web_url = 'https://staging.all-hands.dev'

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', True),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            result = get_cookie_domain()

        assert result is None

    def test_feature_env_returns_none(self):
        """In feature environment, should return None for cookie domain."""
        from server.utils.url_utils import get_cookie_domain

        mock_config = MagicMock()
        mock_config.web_url = 'https://feature-123.staging.all-hands.dev'

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', True),
            patch('server.utils.url_utils.IS_STAGING_ENV', True),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            result = get_cookie_domain()

        assert result is None


class TestGetCookieSamesite:
    """Tests for get_cookie_samesite function."""

    def test_production_with_configured_web_url_returns_strict(self):
        """In production with web_url configured, should return 'strict'.

        We use 'strict' in production for maximum CSRF protection. For invitation
        links from emails, the frontend handles acceptance via an authenticated
        POST request (same-origin), which works with 'strict' cookies.
        """
        from server.utils.url_utils import get_cookie_samesite

        mock_config = MagicMock()
        mock_config.web_url = 'https://app.all-hands.dev'

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', False),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            result = get_cookie_samesite()

        assert result == 'strict'

    def test_production_without_web_url_returns_lax(self):
        """In production without web_url configured, should return 'lax'."""
        from server.utils.url_utils import get_cookie_samesite

        mock_config = MagicMock()
        mock_config.web_url = None

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', False),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            result = get_cookie_samesite()

        assert result == 'lax'

    def test_local_env_returns_lax(self):
        """In local environment, should return 'lax'."""
        from server.utils.url_utils import get_cookie_samesite

        mock_config = MagicMock()
        mock_config.web_url = 'http://localhost:3000'

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', False),
            patch('server.utils.url_utils.IS_LOCAL_ENV', True),
        ):
            result = get_cookie_samesite()

        assert result == 'lax'

    def test_staging_env_returns_lax(self):
        """In staging environment, should return 'lax'."""
        from server.utils.url_utils import get_cookie_samesite

        mock_config = MagicMock()
        mock_config.web_url = 'https://staging.all-hands.dev'

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', True),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            result = get_cookie_samesite()

        assert result == 'lax'

    def test_feature_env_returns_lax(self):
        """In feature environment, should return 'lax'."""
        from server.utils.url_utils import get_cookie_samesite

        mock_config = MagicMock()
        mock_config.web_url = 'https://feature-xyz.staging.all-hands.dev'

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', True),
            patch('server.utils.url_utils.IS_STAGING_ENV', True),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            result = get_cookie_samesite()

        assert result == 'lax'

    def test_empty_web_url_returns_lax(self):
        """Empty web_url should be treated as unconfigured and return 'lax'."""
        from server.utils.url_utils import get_cookie_samesite

        mock_config = MagicMock()
        mock_config.web_url = ''

        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', False),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            result = get_cookie_samesite()

        assert result == 'lax'


class TestSecurityScenarios:
    """Tests for security-critical scenarios."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request object."""
        request = MagicMock()
        request.url = MagicMock()
        return request

    def test_header_poisoning_attack_blocked_when_configured(self, mock_request):
        """
        When web_url is configured, X-Forwarded-* header poisoning should not affect
        the returned URL.
        """
        from server.utils.url_utils import get_web_url

        # Simulate a poisoned request where attacker controls headers
        mock_request.url.hostname = 'evil.com'
        mock_request.url.netloc = 'evil.com:443'

        mock_config = MagicMock()
        mock_config.web_url = 'https://app.all-hands.dev'

        with patch(
            'server.utils.url_utils.get_global_config', return_value=mock_config
        ):
            result = get_web_url(mock_request)

        # Should use configured web_url, not the poisoned request data
        assert result == 'https://app.all-hands.dev'
        assert 'evil' not in result

    def test_cookie_domain_not_set_in_dev_environments(self):
        """
        Cookie domain should not be set in development environments to prevent
        cookies from leaking to other subdomains.
        """
        from server.utils.url_utils import get_cookie_domain

        mock_config = MagicMock()
        mock_config.web_url = 'https://my-feature.staging.all-hands.dev'

        # Test each dev environment
        for env_name, env_config in [
            (
                'local',
                {
                    'IS_LOCAL_ENV': True,
                    'IS_STAGING_ENV': False,
                    'IS_FEATURE_ENV': False,
                },
            ),
            (
                'staging',
                {
                    'IS_LOCAL_ENV': False,
                    'IS_STAGING_ENV': True,
                    'IS_FEATURE_ENV': False,
                },
            ),
            (
                'feature',
                {'IS_LOCAL_ENV': False, 'IS_STAGING_ENV': True, 'IS_FEATURE_ENV': True},
            ),
        ]:
            with (
                patch(
                    'server.utils.url_utils.get_global_config', return_value=mock_config
                ),
                patch(
                    'server.utils.url_utils.IS_FEATURE_ENV',
                    env_config['IS_FEATURE_ENV'],
                ),
                patch(
                    'server.utils.url_utils.IS_STAGING_ENV',
                    env_config['IS_STAGING_ENV'],
                ),
                patch(
                    'server.utils.url_utils.IS_LOCAL_ENV', env_config['IS_LOCAL_ENV']
                ),
            ):
                result = get_cookie_domain()
                assert result is None, f'Expected None for {env_name} environment'

    def test_samesite_strict_in_production_lax_in_dev(self):
        """
        SameSite='strict' should be used in production for maximum CSRF protection.
        SameSite='lax' should be used in dev environments for easier testing.

        For invitation links from emails (cross-site navigation), the frontend
        handles acceptance via a modal that makes an authenticated POST request
        (same-origin), which works with 'strict' cookies.
        """
        from server.utils.url_utils import get_cookie_samesite

        mock_config = MagicMock()
        mock_config.web_url = 'https://app.all-hands.dev'

        # Production should return 'strict'
        with (
            patch('server.utils.url_utils.get_global_config', return_value=mock_config),
            patch('server.utils.url_utils.IS_FEATURE_ENV', False),
            patch('server.utils.url_utils.IS_STAGING_ENV', False),
            patch('server.utils.url_utils.IS_LOCAL_ENV', False),
        ):
            assert get_cookie_samesite() == 'strict'

        # Dev environments should return 'lax'
        for env_config in [
            # Local
            {'IS_LOCAL_ENV': True, 'IS_STAGING_ENV': False, 'IS_FEATURE_ENV': False},
            # Staging
            {'IS_LOCAL_ENV': False, 'IS_STAGING_ENV': True, 'IS_FEATURE_ENV': False},
            # Feature
            {'IS_LOCAL_ENV': False, 'IS_STAGING_ENV': True, 'IS_FEATURE_ENV': True},
        ]:
            with (
                patch(
                    'server.utils.url_utils.get_global_config', return_value=mock_config
                ),
                patch(
                    'server.utils.url_utils.IS_FEATURE_ENV',
                    env_config['IS_FEATURE_ENV'],
                ),
                patch(
                    'server.utils.url_utils.IS_STAGING_ENV',
                    env_config['IS_STAGING_ENV'],
                ),
                patch(
                    'server.utils.url_utils.IS_LOCAL_ENV', env_config['IS_LOCAL_ENV']
                ),
            ):
                assert get_cookie_samesite() == 'lax'
