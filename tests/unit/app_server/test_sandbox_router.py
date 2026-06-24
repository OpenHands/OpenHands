"""Unit tests for the sandbox router endpoints."""

import sys
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Depends, FastAPI, status
from fastapi.testclient import TestClient
from pydantic import BaseModel

# This repository snapshot does not include the agent_server package used by
# app-server modules. Stub the small pieces imported by sandbox_router so this
# focused router regression test can exercise FastAPI parameter binding.
agent_server_module = types.ModuleType('openhands.agent_server')
agent_server_models_module = types.ModuleType('openhands.agent_server.models')
agent_server_utils_module = types.ModuleType('openhands.agent_server.utils')
app_server_config_module = types.ModuleType('openhands.app_server.config')
sandbox_service_module = types.ModuleType(
    'openhands.app_server.sandbox.sandbox_service'
)
session_auth_module = types.ModuleType('openhands.app_server.sandbox.session_auth')
auth_user_context_module = types.ModuleType(
    'openhands.app_server.user.auth_user_context'
)
user_context_module = types.ModuleType('openhands.app_server.user.user_context')
user_auth_module = types.ModuleType('openhands.app_server.user_auth.user_auth')
dependencies_module = types.ModuleType('openhands.app_server.utils.dependencies')


class Success(BaseModel):
    pass


class SandboxService:
    pass


class AuthUserContext:
    pass


class UserContext:
    pass


def check_session_api_key():
    pass


def sandbox_service_dependency():
    pass


agent_server_models_module.Success = Success
agent_server_utils_module.utc_now = lambda: datetime.now(timezone.utc)
app_server_config_module.depends_sandbox_service = lambda: Depends(
    sandbox_service_dependency
)
app_server_config_module.depends_user_context = lambda: Depends(lambda: None)
sandbox_service_module.SandboxService = SandboxService
session_auth_module.validate_session_key = AsyncMock()
auth_user_context_module.AuthUserContext = AuthUserContext
user_context_module.UserContext = UserContext
user_auth_module.get_for_user = AsyncMock()
dependencies_module.check_session_api_key = check_session_api_key
dependencies_module.get_dependencies = lambda: []

sys.modules.setdefault('openhands.agent_server', agent_server_module)
sys.modules.setdefault('openhands.agent_server.models', agent_server_models_module)
sys.modules.setdefault('openhands.agent_server.utils', agent_server_utils_module)
sys.modules.setdefault('openhands.app_server.config', app_server_config_module)
sys.modules.setdefault(
    'openhands.app_server.sandbox.sandbox_service', sandbox_service_module
)
sys.modules.setdefault('openhands.app_server.sandbox.session_auth', session_auth_module)
sys.modules.setdefault(
    'openhands.app_server.user.auth_user_context', auth_user_context_module
)
sys.modules.setdefault('openhands.app_server.user.user_context', user_context_module)
sys.modules.setdefault('openhands.app_server.user_auth.user_auth', user_auth_module)
sys.modules.setdefault('openhands.app_server.utils.dependencies', dependencies_module)

from openhands.app_server.sandbox import sandbox_router  # noqa: E402, I001
from openhands.app_server.utils.dependencies import check_session_api_key  # noqa: E402, I001

router = sandbox_router.router


@pytest.fixture
def mock_sandbox_service():
    """Create a mock SandboxService for router tests."""
    service = MagicMock()
    service.delete_sandbox = AsyncMock(return_value=True)
    return service


@pytest.fixture
def test_client(mock_sandbox_service):
    """Create a test client with the actual sandbox router and mocked service."""
    app = FastAPI()
    app.include_router(router, prefix='/api/v1')

    # Override auth so the test exercises FastAPI routing/parameter binding.
    app.dependency_overrides[check_session_api_key] = lambda: None
    app.dependency_overrides[sandbox_router.sandbox_service_dependency.dependency] = (
        lambda: mock_sandbox_service
    )

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    app.dependency_overrides.clear()


class TestDeleteSandbox:
    """Test suite for the delete_sandbox endpoint."""

    def test_uses_path_id_parameter(self, test_client, mock_sandbox_service):
        """DELETE /sandboxes/{id} binds the path id, not a query parameter."""
        response = test_client.delete('/api/v1/sandboxes/test-sandbox-123')

        assert response.status_code == status.HTTP_200_OK
        mock_sandbox_service.delete_sandbox.assert_awaited_once_with('test-sandbox-123')

    def test_returns_404_when_sandbox_does_not_exist(
        self, test_client, mock_sandbox_service
    ):
        """DELETE /sandboxes/{id} propagates missing sandbox as 404."""
        mock_sandbox_service.delete_sandbox.return_value = False

        response = test_client.delete('/api/v1/sandboxes/missing-sandbox')

        assert response.status_code == status.HTTP_404_NOT_FOUND
        mock_sandbox_service.delete_sandbox.assert_awaited_once_with('missing-sandbox')
