from types import MappingProxyType
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from httpcore import Request
from pydantic import SecretStr

from openhands.app_server.utils.dependencies import check_session_api_key
from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.integrations.service_types import (
    AuthenticationError,
    Repository,
)
from openhands.microagent.types import MicroagentContentResponse
from openhands.server.routes.git import app as git_app
from openhands.server.user_auth import (
    get_access_token,
    get_provider_tokens,
    get_user_id,
)


@pytest.fixture
def test_client():
    """Create a test client for the git API."""
    app = FastAPI()
    app.include_router(git_app)

    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(request: Request, exc: AuthenticationError):
        return JSONResponse(
            status_code=401,
            content=str(exc),
        )

    # Override the FastAPI dependencies directly
    def mock_get_provider_tokens():
        return MappingProxyType(
            {
                ProviderType.GITHUB: ProviderToken(
                    token=SecretStr('ghp_test_token'), host='github.com'
                ),
                ProviderType.GITLAB: ProviderToken(
                    token=SecretStr('glpat_test_token'), host='gitlab.com'
                ),
                ProviderType.BITBUCKET: ProviderToken(
                    token=SecretStr('bb_test_token'), host='bitbucket.org'
                ),
            }
        )

    def mock_get_access_token():
        return None

    def mock_get_user_id():
        return 'test_user'

    def mock_check_session_api_key():
        # Mock session API key check to always pass for tests
        return None

    # Override the dependencies in the app
    app.dependency_overrides[get_provider_tokens] = mock_get_provider_tokens
    app.dependency_overrides[get_access_token] = mock_get_access_token
    app.dependency_overrides[get_user_id] = mock_get_user_id
    app.dependency_overrides[check_session_api_key] = mock_check_session_api_key

    yield TestClient(app)


@pytest.fixture
def mock_github_repository():
    """Create a mock GitHub repository for testing."""
    return Repository(
        id='123456',
        full_name='test/repo',
        git_provider=ProviderType.GITHUB,
        is_public=True,
        stargazers_count=100,
    )


@pytest.fixture
def mock_gitlab_repository():
    """Create a mock GitLab repository for testing."""
    return Repository(
        id='123456',
        full_name='test/repo',
        git_provider=ProviderType.GITLAB,
        is_public=True,
        stargazers_count=100,
    )


@pytest.fixture
def mock_bitbucket_repository():
    """Create a mock Bitbucket repository for testing."""
    return Repository(
        id='123456',
        full_name='test/repo',
        git_provider=ProviderType.BITBUCKET,
        is_public=True,
        stargazers_count=100,
    )


@pytest.fixture
def sample_microagent_content():
    """Sample microagent file content."""
    return """---
name: test_agent
type: repo
inputs:
  - name: query
    type: str
    description: Search query for the repository
mcp_tools:
  stdio_servers:
    - name: git
      command: git
    - name: file_editor
      command: editor
---

This is a test repository microagent for testing purposes."""


@pytest.fixture
def sample_cursorrules_content():
    """Sample .cursorrules file content."""
    return """---
name: cursor_rules
type: repo
---

These are cursor rules for the repository."""


class TestGetRepositoryMicroagents:
    """Test cases for the get_repository_microagents API endpoint."""

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagents_github_success(
        self,
        mock_provider_handler_cls,
        test_client,
        mock_github_repository,
        sample_microagent_content,
        sample_cursorrules_content,
    ):
        """Test successful retrieval of microagents from GitHub repository."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagents method to return sample data
        mock_provider_handler.get_microagents.return_value = [
            {
                'name': 'test_agent',
                'path': '.openhands/microagents/test_agent.md',
                'created_at': '2024-01-01T00:00:00',
            },
            {
                'name': 'cursorrules',
                'path': '.cursorrules',
                'created_at': '2024-01-01T00:00:00',
            },
        ]

        # Execute test
        response = test_client.get('/api/user/repository/test/repo/microagents')

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2  # .cursorrules + 1 .md file

        # Check that basic fields are present (content is excluded for performance)
        for microagent in data:
            assert 'name' in microagent
            assert 'path' in microagent
            assert 'created_at' in microagent
            # Content field should not be present in listing API
            assert 'content' not in microagent
            # Type and other detailed fields are no longer included in listing API
            assert 'type' not in microagent
            assert 'triggers' not in microagent
            assert 'inputs' not in microagent
            assert 'tools' not in microagent

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagents_gitlab_success(
        self,
        mock_provider_handler_cls,
        test_client,
        mock_gitlab_repository,
    ):
        """Test successful retrieval of microagents from GitLab repository."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagents method to return sample data
        mock_provider_handler.get_microagents.return_value = [
            {
                'name': 'test_agent',
                'path': '.openhands/microagents/test_agent.md',
                'created_at': '2024-01-01T00:00:00',
            }
        ]

        # Execute test
        response = test_client.get('/api/user/repository/test/repo/microagents')

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1  # Only 1 .md file
        assert 'content' not in data[0]  # Content should not be present in listing API

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagents_bitbucket_success(
        self,
        mock_provider_handler_cls,
        test_client,
        mock_bitbucket_repository,
    ):
        """Test successful retrieval of microagents from Bitbucket repository."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagents method to return sample data
        mock_provider_handler.get_microagents.return_value = [
            {
                'name': 'test_agent',
                'path': '.openhands/microagents/test_agent.md',
                'created_at': '2024-01-01T00:00:00',
            }
        ]

        # Execute test
        response = test_client.get('/api/user/repository/test/repo/microagents')

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1  # Only 1 .md file
        assert 'content' not in data[0]  # Content should not be present in listing API

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagents_no_directory_found(
        self,
        mock_provider_handler_cls,
        test_client,
        mock_github_repository,
    ):
        """Test when microagents directory is not found."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagents method to return empty list
        mock_provider_handler.get_microagents.return_value = []

        # Execute test
        response = test_client.get('/api/user/repository/test/repo/microagents')

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data == []

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagents_authentication_error(
        self,
        mock_provider_handler_cls,
        test_client,
    ):
        """Test authentication error."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagents method to raise AuthenticationError
        mock_provider_handler.get_microagents.side_effect = AuthenticationError(
            'Invalid credentials'
        )

        # Execute test
        response = test_client.get('/api/user/repository/test/repo/microagents')

        # Assertions
        assert response.status_code == 401
        assert response.json() == 'Invalid credentials'


class TestGetRepositoryMicroagentContent:
    """Test cases for the get_repository_microagent_content API endpoint."""

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagent_content_github_success(
        self,
        mock_provider_handler_cls,
        test_client,
        sample_microagent_content,
    ):
        """Test successful retrieval of microagent content from GitHub."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagent_content method
        mock_provider_handler.get_microagent_content.return_value = (
            MicroagentContentResponse(
                content=sample_microagent_content,
                path='.openhands/microagents/test_agent.md',
                triggers=['test', 'agent'],
            )
        )

        # Execute test
        file_path = '.openhands/microagents/test_agent.md'
        response = test_client.get(
            f'/api/user/repository/test/repo/microagents/content?file_path={quote(file_path)}'
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert 'content' in data
        assert data['content'] == sample_microagent_content
        assert data['path'] == file_path
        assert 'triggers' in data
        assert data['triggers'] == ['test', 'agent']

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagent_content_gitlab_success(
        self,
        mock_provider_handler_cls,
        test_client,
        sample_microagent_content,
    ):
        """Test successful retrieval of microagent content from GitLab."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagent_content method
        mock_provider_handler.get_microagent_content.return_value = (
            MicroagentContentResponse(
                content=sample_microagent_content,
                path='.openhands/microagents/test_agent.md',
                triggers=['test', 'agent'],
            )
        )

        # Execute test
        file_path = '.openhands/microagents/test_agent.md'
        response = test_client.get(
            f'/api/user/repository/test/repo/microagents/content?file_path={quote(file_path)}'
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data['content'] == sample_microagent_content
        assert data['path'] == file_path
        assert 'triggers' in data
        assert data['triggers'] == ['test', 'agent']

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagent_content_bitbucket_success(
        self,
        mock_provider_handler_cls,
        test_client,
        sample_microagent_content,
    ):
        """Test successful retrieval of microagent content from Bitbucket."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagent_content method
        mock_provider_handler.get_microagent_content.return_value = (
            MicroagentContentResponse(
                content=sample_microagent_content,
                path='.openhands/microagents/test_agent.md',
                triggers=['test', 'agent'],
            )
        )

        # Execute test
        file_path = '.openhands/microagents/test_agent.md'
        response = test_client.get(
            f'/api/user/repository/test/repo/microagents/content?file_path={quote(file_path)}'
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data['content'] == sample_microagent_content
        assert data['path'] == file_path
        assert 'triggers' in data
        assert data['triggers'] == ['test', 'agent']

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagent_content_file_not_found(
        self,
        mock_provider_handler_cls,
        test_client,
        mock_github_repository,
    ):
        """Test when microagent file is not found."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagent_content method to raise RuntimeError
        mock_provider_handler.get_microagent_content.side_effect = RuntimeError(
            'File not found'
        )

        # Execute test
        file_path = '.openhands/microagents/nonexistent.md'
        response = test_client.get(
            f'/api/user/repository/test/repo/microagents/content?file_path={quote(file_path)}'
        )

        # Assertions
        assert response.status_code == 500
        assert 'File not found' in response.json()

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagent_content_authentication_error(
        self,
        mock_provider_handler_cls,
        test_client,
    ):
        """Test authentication error for content API."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagent_content method to raise AuthenticationError
        mock_provider_handler.get_microagent_content.side_effect = AuthenticationError(
            'Invalid credentials'
        )

        # Execute test
        file_path = '.openhands/microagents/test_agent.md'
        response = test_client.get(
            f'/api/user/repository/test/repo/microagents/content?file_path={quote(file_path)}'
        )

        # Assertions
        assert response.status_code == 401
        assert response.json() == 'Invalid credentials'

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagent_content_cursorrules(
        self,
        mock_provider_handler_cls,
        test_client,
        sample_cursorrules_content,
    ):
        """Test retrieval of .cursorrules file content."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagent_content method
        mock_provider_handler.get_microagent_content.return_value = (
            MicroagentContentResponse(
                content=sample_cursorrules_content,
                path='.cursorrules',
                triggers=['cursor', 'rules'],
            )
        )

        # Execute test
        file_path = '.cursorrules'
        response = test_client.get(
            f'/api/user/repository/test/repo/microagents/content?file_path={quote(file_path)}'
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data['content'] == sample_cursorrules_content
        assert data['path'] == file_path
        assert 'triggers' in data
        assert data['triggers'] == ['cursor', 'rules']


class TestSpecialRepositoryStructures:
    """Test cases for special repository structures."""

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagents_openhands_repo_structure(
        self,
        mock_provider_handler_cls,
        test_client,
    ):
        """Test microagents from .openhands repository structure."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagents method to return sample data for .openhands repo
        mock_provider_handler.get_microagents.return_value = [
            {
                'name': 'test_agent',
                'path': 'microagents/test_agent.md',  # Should be in microagents folder, not .openhands/microagents
                'created_at': '2024-01-01T00:00:00',
            }
        ]

        # Execute test
        response = test_client.get('/api/user/repository/test/.openhands/microagents')

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert (
            data[0]['path'] == 'microagents/test_agent.md'
        )  # Should be in microagents folder, not .openhands/microagents

    @pytest.mark.asyncio
    @patch('openhands.server.routes.git.ProviderHandler')
    async def test_get_microagents_gitlab_openhands_config_structure(
        self,
        mock_provider_handler_cls,
        test_client,
    ):
        """Test microagents from GitLab openhands-config repository structure."""
        # Setup mocks
        mock_provider_handler = AsyncMock()
        mock_provider_handler_cls.return_value = mock_provider_handler

        # Mock the get_microagents method to return sample data for openhands-config repo
        mock_provider_handler.get_microagents.return_value = [
            {
                'name': 'test_agent',
                'path': 'microagents/test_agent.md',  # Should be in microagents folder, not .openhands/microagents
                'created_at': '2024-01-01T00:00:00',
            }
        ]

        # Execute test
        response = test_client.get(
            '/api/user/repository/test/openhands-config/microagents'
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert (
            data[0]['path'] == 'microagents/test_agent.md'
        )  # Should be in microagents folder, not .openhands/microagents
