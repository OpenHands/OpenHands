"""LLM diagnostics and health check endpoints."""

import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends

from openhands.app_server.utils.dependencies import get_dependencies
from openhands.core.config import config
from openhands.core.config.llm_config import LLMConfig
from openhands.server.user_auth import get_user_settings
from openhands.storage.data_models.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/llm', tags=['llm'], dependencies=get_dependencies())


class LLMHealthCheckResponse:
    """Response model for LLM health check."""

    def __init__(
        self,
        status: str,
        model: str,
        provider: str,
        latency_ms: float = 0,
        error_message: str = '',
        is_local: bool = False,
    ):
        self.status = status
        self.model = model
        self.provider = provider
        self.latency_ms = latency_ms
        self.error_message = error_message
        self.is_local = is_local

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            'status': self.status,
            'model': self.model,
            'provider': self.provider,
            'latency_ms': round(self.latency_ms, 2),
            'error_message': self.error_message,
            'is_local': self.is_local,
        }


async def _test_llm_connection(llm_config: LLMConfig) -> LLMHealthCheckResponse:
    """Test LLM connection with timeout handling.

    Args:
        llm_config: LLM configuration to test

    Returns:
        LLMHealthCheckResponse with connection status
    """
    start_time = time.time()
    model = llm_config.model or 'unknown'
    provider = llm_config.custom_llm_provider or model.split('/', maxsplit=1)[0]

    # Determine if local model
    is_local = False
    if llm_config.base_url:
        for substring in ['localhost', '127.0.0.1', '0.0.0.0', 'host.docker.internal']:
            if substring in llm_config.base_url:
                is_local = True
                break
    elif model.startswith('ollama/'):
        is_local = True

    # Special handling for Ollama
    if model.startswith('ollama/'):
        ollama_base_url = (
            llm_config.ollama_base_url
            or llm_config.base_url
            or 'http://localhost:11434'
        )
        ollama_url = ollama_base_url.strip('/') + '/api/tags'

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(ollama_url)
                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    models_data = response.json()
                    # Check if our model is in the list
                    model_name = model.replace('ollama/', '')
                    available_models = [
                        m.get('name', '') for m in models_data.get('models', [])
                    ]

                    if model_name in available_models:
                        return LLMHealthCheckResponse(
                            status='connected',
                            model=model,
                            provider='ollama',
                            latency_ms=latency_ms,
                            is_local=True,
                        )
                    else:
                        return LLMHealthCheckResponse(
                            status='model_not_found',
                            model=model,
                            provider='ollama',
                            latency_ms=latency_ms,
                            error_message='Model not found in Ollama. '
                            + f'Available: {", ".join(available_models[:5])}',
                            is_local=True,
                        )
                else:
                    return LLMHealthCheckResponse(
                        status='error',
                        model=model,
                        provider='ollama',
                        latency_ms=(time.time() - start_time) * 1000,
                        error_message=f'Ollama returned status {response.status_code}',
                        is_local=True,
                    )
        except httpx.TimeoutException:
            latency_ms = (time.time() - start_time) * 1000
            return LLMHealthCheckResponse(
                status='timeout',
                model=model,
                provider='ollama',
                latency_ms=latency_ms,
                error_message='Connection timeout after '
                + f'{latency_ms:.0f}ms. Is Ollama running at {ollama_base_url}?',
                is_local=True,
            )
        except httpx.ConnectError as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMHealthCheckResponse(
                status='connection_error',
                model=model,
                provider='ollama',
                latency_ms=latency_ms,
                error_message='Cannot connect to Ollama at '
                + f'{ollama_base_url}. Error: {str(e)[:100]}',
                is_local=True,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMHealthCheckResponse(
                status='error',
                model=model,
                provider='ollama',
                latency_ms=latency_ms,
                error_message=f'Ollama health check failed: {str(e)[:100]}',
                is_local=True,
            )

    # For remote models (OpenAI, Anthropic, etc.), we can't easily test without credentials
    # So we return a status indicating we can't verify but settings look valid
    return LLMHealthCheckResponse(
        status='configured',
        model=model,
        provider=provider,
        latency_ms=0,
        error_message='Remote models cannot be fully tested. Verify API credentials in settings.',
        is_local=False,
    )


@router.get('/health-check')
async def llm_health_check(
    settings: Settings = Depends(get_user_settings),
) -> dict[str, Any]:
    """Check health and connectivity of configured LLM.

    Returns:
        Health status with latency and error details
    """
    try:
        # Get LLM config
        llm_config = config.llm

        # If model is set in settings, use that
        if settings.llm_model:
            llm_config = LLMConfig(
                model=settings.llm_model,
                api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
                base_url=settings.llm_base_url or llm_config.base_url,
                custom_llm_provider=llm_config.custom_llm_provider,
                ollama_base_url=llm_config.ollama_base_url,
            )

        response = await _test_llm_connection(llm_config)
        return response.to_dict()

    except Exception as e:
        logger.exception('Error checking LLM health')
        return {
            'status': 'error',
            'model': 'unknown',
            'provider': 'unknown',
            'latency_ms': 0,
            'error_message': f'Health check failed: {str(e)[:100]}',
            'is_local': False,
        }
