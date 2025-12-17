"""Unit tests for the V1 skills endpoint in app_conversation_router.

This module tests the GET /{conversation_id}/skills endpoint functionality,
following TDD best practices with AAA structure.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
)
from openhands.app_server.app_conversation.app_conversation_router import (
    get_conversation_skills,
)
from openhands.app_server.app_conversation.app_conversation_service_base import (
    AppConversationServiceBase,
)
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    ExposedUrl,
    SandboxInfo,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.user.user_context import UserContext
from openhands.sdk.context.skills import KeywordTrigger, Skill, TaskTrigger


def _make_service_mock(
    *,
    user_context: UserContext,
    conversation_return: AppConversation | None = None,
    skills_return: list[Skill] | None = None,
    raise_on_load: bool = False,
):
    """Create a mock service that passes the isinstance check and returns the desired values."""

    mock_cls = type('AppConversationServiceMock', (MagicMock,), {})
    AppConversationServiceBase.register(mock_cls)

    service = mock_cls()
    service.user_context = user_context
    service.get_app_conversation = AsyncMock(return_value=conversation_return)

    async def _load_skills(*_args, **_kwargs):
        if raise_on_load:
            raise Exception('Skill loading failed')
        return skills_return or []

    service.load_and_merge_all_skills = AsyncMock(side_effect=_load_skills)
    return service


@pytest.mark.asyncio
class TestGetConversationSkills:
    """Test suite for get_conversation_skills endpoint."""

    async def test_get_skills_returns_repo_and_knowledge_skills(self):
        """Test successful retrieval of both repo and knowledge skills.

        Arrange: Setup conversation, sandbox, and skills with different types
        Act: Call get_conversation_skills endpoint
        Assert: Response contains both repo and knowledge skills with correct types
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        working_dir = '/workspace'

        # Create mock conversation
        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            selected_repository='owner/repo',
            sandbox_status=SandboxStatus.RUNNING,
        )

        # Create mock sandbox with agent server URL
        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            created_by_user_id='test-user',
            status=SandboxStatus.RUNNING,
            sandbox_spec_id=str(uuid4()),
            session_api_key='test-api-key',
            exposed_urls=[
                ExposedUrl(name=AGENT_SERVER, url='http://localhost:8000', port=8000)
            ],
        )

        # Create mock sandbox spec
        mock_sandbox_spec = SandboxSpecInfo(
            id=str(uuid4()), command=None, working_dir=working_dir
        )

        # Create mock skills - repo skill (no trigger)
        repo_skill = Skill(
            name='repo_skill',
            content='Repository skill content',
            trigger=None,
        )

        # Create mock skills - knowledge skill (with KeywordTrigger)
        knowledge_skill = Skill(
            name='knowledge_skill',
            content='Knowledge skill content',
            trigger=KeywordTrigger(keywords=['test', 'help']),
        )

        # Mock services
        mock_user_context = MagicMock(spec=UserContext)
        mock_app_conversation_service = _make_service_mock(
            user_context=mock_user_context,
            conversation_return=mock_conversation,
            skills_return=[repo_skill, knowledge_skill],
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=mock_sandbox)

        mock_sandbox_spec_service = MagicMock()
        mock_sandbox_spec_service.get_sandbox_spec = AsyncMock(
            return_value=mock_sandbox_spec
        )

        # Act
        response = await get_conversation_skills(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK
        content = response.body.decode('utf-8')
        import json

        data = json.loads(content)
        assert 'skills' in data
        assert len(data['skills']) == 2

        # Check repo skill
        repo_skill_data = next(
            (s for s in data['skills'] if s['name'] == 'repo_skill'), None
        )
        assert repo_skill_data is not None
        assert repo_skill_data['type'] == 'repo'
        assert repo_skill_data['content'] == 'Repository skill content'
        assert repo_skill_data['triggers'] == []

        # Check knowledge skill
        knowledge_skill_data = next(
            (s for s in data['skills'] if s['name'] == 'knowledge_skill'), None
        )
        assert knowledge_skill_data is not None
        assert knowledge_skill_data['type'] == 'knowledge'
        assert knowledge_skill_data['content'] == 'Knowledge skill content'
        assert knowledge_skill_data['triggers'] == ['test', 'help']

    async def test_get_skills_returns_404_when_conversation_not_found(self):
        """Test endpoint returns 404 when conversation doesn't exist.

        Arrange: Setup mocks to return None for conversation
        Act: Call get_conversation_skills endpoint
        Assert: Response is 404 with appropriate error message
        """
        # Arrange
        conversation_id = uuid4()

        mock_user_context = MagicMock(spec=UserContext)
        mock_app_conversation_service = _make_service_mock(
            user_context=mock_user_context,
            conversation_return=None,
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_spec_service = MagicMock()

        # Act
        response = await get_conversation_skills(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
        )

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
        content = response.body.decode('utf-8')
        import json

        data = json.loads(content)
        assert 'error' in data
        assert str(conversation_id) in data['error']

    async def test_get_skills_returns_404_when_sandbox_not_found(self):
        """Test endpoint returns 404 when sandbox doesn't exist.

        Arrange: Setup conversation but no sandbox
        Act: Call get_conversation_skills endpoint
        Assert: Response is 404 with sandbox error message
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())

        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            sandbox_status=SandboxStatus.RUNNING,
        )

        mock_user_context = MagicMock(spec=UserContext)
        mock_app_conversation_service = _make_service_mock(
            user_context=mock_user_context,
            conversation_return=mock_conversation,
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=None)

        mock_sandbox_spec_service = MagicMock()

        # Act
        response = await get_conversation_skills(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
        )

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
        content = response.body.decode('utf-8')
        import json

        data = json.loads(content)
        assert 'error' in data
        assert 'Sandbox not found' in data['error']

    async def test_get_skills_returns_404_when_sandbox_not_running(self):
        """Test endpoint returns 404 when sandbox is not in RUNNING state.

        Arrange: Setup conversation with stopped sandbox
        Act: Call get_conversation_skills endpoint
        Assert: Response is 404 with sandbox not running message
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())

        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            sandbox_status=SandboxStatus.PAUSED,
        )

        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            created_by_user_id='test-user',
            status=SandboxStatus.PAUSED,
            sandbox_spec_id=str(uuid4()),
            session_api_key='test-api-key',
        )

        mock_user_context = MagicMock(spec=UserContext)
        mock_app_conversation_service = _make_service_mock(
            user_context=mock_user_context,
            conversation_return=mock_conversation,
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=mock_sandbox)

        mock_sandbox_spec_service = MagicMock()

        # Act
        response = await get_conversation_skills(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
        )

        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND
        content = response.body.decode('utf-8')
        import json

        data = json.loads(content)
        assert 'error' in data
        assert 'not running' in data['error']

    async def test_get_skills_handles_task_trigger_skills(self):
        """Test endpoint correctly handles skills with TaskTrigger.

        Arrange: Setup skill with TaskTrigger
        Act: Call get_conversation_skills endpoint
        Assert: Skill is categorized as knowledge type with correct triggers
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())

        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            sandbox_status=SandboxStatus.RUNNING,
        )

        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            created_by_user_id='test-user',
            status=SandboxStatus.RUNNING,
            sandbox_spec_id=str(uuid4()),
            session_api_key='test-api-key',
            exposed_urls=[
                ExposedUrl(name=AGENT_SERVER, url='http://localhost:8000', port=8000)
            ],
        )

        mock_sandbox_spec = SandboxSpecInfo(
            id=str(uuid4()), command=None, working_dir='/workspace'
        )

        # Create task skill with TaskTrigger
        task_skill = Skill(
            name='task_skill',
            content='Task skill content',
            trigger=TaskTrigger(triggers=['task', 'execute']),
        )

        mock_user_context = MagicMock(spec=UserContext)
        mock_app_conversation_service = _make_service_mock(
            user_context=mock_user_context,
            conversation_return=mock_conversation,
            skills_return=[task_skill],
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=mock_sandbox)

        mock_sandbox_spec_service = MagicMock()
        mock_sandbox_spec_service.get_sandbox_spec = AsyncMock(
            return_value=mock_sandbox_spec
        )

        # Act
        response = await get_conversation_skills(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK
        content = response.body.decode('utf-8')
        import json

        data = json.loads(content)
        assert len(data['skills']) == 1
        skill_data = data['skills'][0]
        assert skill_data['type'] == 'knowledge'
        assert skill_data['triggers'] == ['task', 'execute']

    async def test_get_skills_returns_500_on_skill_loading_error(self):
        """Test endpoint returns 500 when skill loading fails.

        Arrange: Setup mocks to raise exception during skill loading
        Act: Call get_conversation_skills endpoint
        Assert: Response is 500 with error message
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())

        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            sandbox_status=SandboxStatus.RUNNING,
        )

        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            created_by_user_id='test-user',
            status=SandboxStatus.RUNNING,
            sandbox_spec_id=str(uuid4()),
            session_api_key='test-api-key',
            exposed_urls=[
                ExposedUrl(name=AGENT_SERVER, url='http://localhost:8000', port=8000)
            ],
        )

        mock_sandbox_spec = SandboxSpecInfo(
            id=str(uuid4()), command=None, working_dir='/workspace'
        )

        mock_user_context = MagicMock(spec=UserContext)
        mock_app_conversation_service = _make_service_mock(
            user_context=mock_user_context,
            conversation_return=mock_conversation,
            raise_on_load=True,
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=mock_sandbox)

        mock_sandbox_spec_service = MagicMock()
        mock_sandbox_spec_service.get_sandbox_spec = AsyncMock(
            return_value=mock_sandbox_spec
        )

        # Act
        response = await get_conversation_skills(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
        )

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        content = response.body.decode('utf-8')
        import json

        data = json.loads(content)
        assert 'error' in data
        assert 'Error getting skills' in data['error']

    async def test_get_skills_returns_empty_list_when_no_skills_loaded(self):
        """Test endpoint returns empty skills list when no skills are found.

        Arrange: Setup all skill loaders to return empty lists
        Act: Call get_conversation_skills endpoint
        Assert: Response contains empty skills array
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())

        mock_conversation = AppConversation(
            id=conversation_id,
            created_by_user_id='test-user',
            sandbox_id=sandbox_id,
            sandbox_status=SandboxStatus.RUNNING,
        )

        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            created_by_user_id='test-user',
            status=SandboxStatus.RUNNING,
            sandbox_spec_id=str(uuid4()),
            session_api_key='test-api-key',
            exposed_urls=[
                ExposedUrl(name=AGENT_SERVER, url='http://localhost:8000', port=8000)
            ],
        )

        mock_sandbox_spec = SandboxSpecInfo(
            id=str(uuid4()), command=None, working_dir='/workspace'
        )

        mock_user_context = MagicMock(spec=UserContext)
        mock_app_conversation_service = _make_service_mock(
            user_context=mock_user_context,
            conversation_return=mock_conversation,
            skills_return=[],
        )

        mock_sandbox_service = MagicMock()
        mock_sandbox_service.get_sandbox = AsyncMock(return_value=mock_sandbox)

        mock_sandbox_spec_service = MagicMock()
        mock_sandbox_spec_service.get_sandbox_spec = AsyncMock(
            return_value=mock_sandbox_spec
        )

        # Act
        response = await get_conversation_skills(
            conversation_id=conversation_id,
            app_conversation_service=mock_app_conversation_service,
            sandbox_service=mock_sandbox_service,
            sandbox_spec_service=mock_sandbox_spec_service,
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK
        content = response.body.decode('utf-8')
        import json

        data = json.loads(content)
        assert 'skills' in data
        assert len(data['skills']) == 0
