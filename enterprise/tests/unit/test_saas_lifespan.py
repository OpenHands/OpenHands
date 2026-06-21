"""Tests for SaasAppLifespanService."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_analytics_service():
    svc = MagicMock()
    svc.shutdown = MagicMock()
    return svc


@pytest.mark.asyncio
async def test_aenter_calls_init_analytics_service():
    """SaasAppLifespanService.__aenter__ initializes the analytics service."""
    from server.app_lifespan.saas_app_lifespan_service import SaasAppLifespanService

    with patch(
        'server.app_lifespan.saas_app_lifespan_service.init_analytics_service'
    ) as mock_init:
        svc = SaasAppLifespanService()
        await svc.__aenter__()
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_aenter_passes_env_vars_to_init():
    """SaasAppLifespanService reads config from env vars."""
    from server.app_lifespan.saas_app_lifespan_service import SaasAppLifespanService

    with (
        patch(
            'server.app_lifespan.saas_app_lifespan_service.init_analytics_service'
        ) as mock_init,
        patch.dict(
            'os.environ',
            {
                'POSTHOG_CLIENT_KEY': 'test-key',
                'POSTHOG_HOST': 'https://test.posthog.com',
                'OPENHANDS_CONFIG_CLS': 'enterprise.server.config.SaaSServerConfig',
            },
        ),
    ):
        svc = SaasAppLifespanService()
        await svc.__aenter__()

        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs['api_key'] == 'test-key'
        assert call_kwargs.kwargs['host'] == 'https://test.posthog.com'


@pytest.mark.asyncio
async def test_aexit_calls_shutdown_when_service_exists(mock_analytics_service):
    """SaasAppLifespanService.__aexit__ calls shutdown on the analytics service."""
    from server.app_lifespan.saas_app_lifespan_service import SaasAppLifespanService

    with (
        patch('server.app_lifespan.saas_app_lifespan_service.init_analytics_service'),
        patch(
            'server.app_lifespan.saas_app_lifespan_service.get_analytics_service',
            return_value=mock_analytics_service,
        ),
    ):
        svc = SaasAppLifespanService()
        await svc.__aenter__()
        await svc.__aexit__(None, None, None)

        mock_analytics_service.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_aexit_does_not_raise_when_service_is_none():
    """SaasAppLifespanService.__aexit__ does not raise if analytics service is None."""
    from server.app_lifespan.saas_app_lifespan_service import SaasAppLifespanService

    with (
        patch('server.app_lifespan.saas_app_lifespan_service.init_analytics_service'),
        patch(
            'server.app_lifespan.saas_app_lifespan_service.get_analytics_service',
            return_value=None,
        ),
    ):
        svc = SaasAppLifespanService()
        await svc.__aenter__()
        # Must not raise
        await svc.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_aexit_does_not_raise_on_shutdown_error(mock_analytics_service):
    """SaasAppLifespanService.__aexit__ swallows errors from shutdown."""
    from server.app_lifespan.saas_app_lifespan_service import SaasAppLifespanService

    mock_analytics_service.shutdown.side_effect = RuntimeError('connection closed')

    with (
        patch('server.app_lifespan.saas_app_lifespan_service.init_analytics_service'),
        patch(
            'server.app_lifespan.saas_app_lifespan_service.get_analytics_service',
            return_value=mock_analytics_service,
        ),
    ):
        svc = SaasAppLifespanService()
        await svc.__aenter__()
        # Must not raise even if shutdown errors
        await svc.__aexit__(None, None, None)
