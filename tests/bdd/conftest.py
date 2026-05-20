"""Main pytest configuration for BDD tests.

Provides fixtures for:
- Mock LLM service
- Mock sandbox with filesystem
- App server with mocked dependencies
- HTTP client for API calls
- Playwright browser for E2E testing
- Logging configuration
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

import pytest
from playwright.async_api import Browser, async_playwright

from tests.bdd.mocks.llm_mock import LLMMock
from tests.bdd.mocks.sandbox_mock import MockSandbox

# Import step modules for pytest-bdd discovery
from tests.bdd.steps import (
    agent_steps,  # noqa: F401
    common_steps,  # noqa: F401
    frontend_steps,  # noqa: F401
    skills_steps,  # noqa: F401
)
from tests.bdd.utils.api_client import AppServerAPIClient
from tests.bdd.utils.test_data import TEST_APP_SERVER_URL


# Configure logging for BDD tests
def configure_logging() -> None:
    """Configure logging for tests.

    Logs go to both console and file (logs/bdd-tests.log).
    """
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'bdd-tests.log'

    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_handler.setFormatter(console_format)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler.setFormatter(file_format)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Log test start
    logger = logging.getLogger(__name__)
    logger.info('=' * 80)
    logger.info('BDD Test Suite Started')
    logger.info(f'Logs written to: {log_file}')
    logger.info('=' * 80)


# Configure logging on module import
configure_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures: Mock Services
# ============================================================================


@pytest.fixture
def mock_llm() -> Generator[LLMMock, None, None]:
    """Provide mock LLM service.

    Returns:
        LLMMock instance (reset after each test)
    """
    llm = LLMMock()
    logger.debug('Created mock LLM')
    yield llm
    logger.debug(f'Mock LLM stats: {llm.get_stats()}')
    llm.reset()


@pytest.fixture
def mock_sandbox() -> Generator[MockSandbox, None, None]:
    """Provide mock sandbox with filesystem.

    Returns:
        MockSandbox instance (reset after each test)
    """
    sandbox = MockSandbox()
    logger.debug(
        f'Created mock sandbox with filesystem: {sandbox.filesystem.root_path}'
    )
    yield sandbox
    logger.debug(f'Mock sandbox stats: {sandbox.get_stats()}')
    sandbox.reset()


# ============================================================================
# Fixtures: HTTP Client
# ============================================================================


@pytest.fixture
async def http_client() -> AsyncGenerator[AppServerAPIClient, None]:
    """Provide HTTP client for app-server API.

    Returns:
        AppServerAPIClient instance
    """
    client = AppServerAPIClient(TEST_APP_SERVER_URL)
    async with client:
        logger.debug(f'Created HTTP client pointing to {TEST_APP_SERVER_URL}')
        yield client


# ============================================================================
# Fixtures: Browser (Playwright)
# ============================================================================


@pytest.fixture
async def browser() -> AsyncGenerator[Browser, None]:
    """Provide Playwright browser instance.

    Returns:
        Playwright Browser object
    """
    async with async_playwright() as p:
        # Check for --headed flag in pytest arguments
        headed = '--headed' in sys.argv

        browser_instance = await p.chromium.launch(
            headless=not headed,
            args=['--disable-blink-features=AutomationControlled'],
        )

        logger.debug(f'Launched Chromium browser (headless={not headed})')
        yield browser_instance

        await browser_instance.close()
        logger.debug('Closed Chromium browser')


# ============================================================================
# Fixtures: Test Configuration
# ============================================================================


@pytest.fixture
def bdd_config() -> dict[str, Any]:
    """Provide BDD test configuration.

    Returns:
        Config dict with test parameters
    """
    return {
        'app_server_url': TEST_APP_SERVER_URL,
        'app_server_port': 9999,
        'browser_timeout': 5.0,
        'api_timeout': 30.0,
        'test_user_id': 'test-user-123',
        'test_conversation_id': 'test-conv-001',
    }


# ============================================================================
# Fixtures: Async Event Loop
# ============================================================================


@pytest.fixture(scope='session')
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Provide event loop for async tests.

    Scope: session (single loop for all async tests)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.debug('Created session event loop')
    yield loop
    loop.close()
    logger.debug('Closed session event loop')


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config: Any) -> None:
    """Configure pytest.

    Args:
        config: Pytest config object
    """
    # Register custom markers
    config.addinivalue_line(
        'markers',
        'fast: fast tests with mocked services',
    )
    config.addinivalue_line(
        'markers',
        'slow: slow tests with real services',
    )
    config.addinivalue_line(
        'markers',
        'regression: regression tests',
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Modify test collection.

    Args:
        config: Pytest config
        items: Collected test items
    """
    # Mark all tests as fast by default unless they have @slow marker
    for item in items:
        if 'slow' not in item.keywords:
            item.add_marker(pytest.mark.fast)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    """Log test results.

    Args:
        item: Test item
        call: Test call context
    """
    outcome = yield

    if call.when == 'call':
        report = outcome.get_result()
        test_name = item.name
        if report.passed:
            logger.debug(f'✓ PASSED: {test_name}')
        elif report.failed:
            logger.error(f'✗ FAILED: {test_name}')
        elif report.skipped:
            logger.info(f'⊘ SKIPPED: {test_name}')


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def test_logger(request: Any) -> Generator[logging.Logger, None, None]:
    """Provide logger for individual tests.

    Automatically used by all tests.

    Args:
        request: Pytest request object

    Yields:
        Logger instance
    """
    test_logger_instance = logging.getLogger(request.node.name)
    test_logger_instance.info(f'Starting: {request.node.name}')
    yield test_logger_instance
    test_logger_instance.info(f'Finished: {request.node.name}')


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Provide temporary workspace directory.

    Args:
        tmp_path: Pytest tmp_path fixture

    Returns:
        Temporary directory path
    """
    logger.debug(f'Created temporary workspace: {tmp_path}')
    return tmp_path


# ============================================================================
# BDD Test Context Fixtures
# ============================================================================


class AgentContext:
    """Shared context for agent test steps."""

    def __init__(self) -> None:
        """Initialize context."""
        self.llm: Any | None = None
        self.messages: list[dict[str, Any]] = []
        self.last_response: dict[str, Any] | None = None
        self.llm_call_count: int = 0

    def reset(self) -> None:
        """Reset context."""
        self.messages.clear()
        self.last_response = None
        self.llm_call_count = 0


@pytest.fixture
def agent_context() -> Generator[AgentContext, None, None]:
    """Provide agent context for steps.

    Returns:
        AgentContext instance (reset after test)
    """
    context = AgentContext()
    logger.debug('Created agent context')
    yield context
    logger.debug(
        f'Agent context stats: messages={len(context.messages)}, calls={context.llm_call_count}'
    )
    context.reset()


# ============================================================================
# Skills Testing Fixtures
# ============================================================================


class EnvVarController:
    """Control environment variables for testing."""

    def __init__(self) -> None:
        """Initialize controller."""
        self._original_values: dict[str, str | None] = {}

    def set(self, name: str, value: str) -> None:
        """Set environment variable.

        Args:
            name: Variable name
            value: Variable value
        """
        if name not in self._original_values:
            self._original_values[name] = os.environ.get(name)
        os.environ[name] = value
        logger.debug(f'Set {name}={value}')

    def get(self, name: str) -> str | None:
        """Get environment variable.

        Args:
            name: Variable name

        Returns:
            Variable value or None
        """
        return os.environ.get(name)

    def unset(self, name: str) -> None:
        """Unset environment variable.

        Args:
            name: Variable name
        """
        if name not in self._original_values:
            self._original_values[name] = os.environ.get(name)
        if name in os.environ:
            del os.environ[name]
        logger.debug(f'Unset {name}')

    def restore(self) -> None:
        """Restore original values."""
        for name, value in self._original_values.items():
            if value is None:
                if name in os.environ:
                    del os.environ[name]
            else:
                os.environ[name] = value
        self._original_values.clear()
        logger.debug('Restored original env vars')


@pytest.fixture
def env_var_controller() -> Generator[EnvVarController, None, None]:
    """Provide environment variable controller.

    Returns:
        EnvVarController instance (restores original values after test)
    """
    controller = EnvVarController()
    logger.debug('Created env var controller')
    yield controller
    logger.debug('Restoring env vars')
    controller.restore()


class SkillsTestEnvironment:
    """Test environment for skills loading tests.

    Coordinates:
    - Mock agent-server /api/skills endpoint
    - Skill configuration
    - Conversation creation
    - Skills assertion
    """

    def __init__(
        self,
        mock_agent_server_skills: Any,
    ) -> None:
        """Initialize environment.

        Args:
            mock_agent_server_skills: Mock agent-server API
        """
        self.skills_api = mock_agent_server_skills
        self.agent = None
        self.last_conversation = None
        self.current_project = None
        self.conversations: dict[str, Any] = {}
        self.project_a_skills: list[str] = []
        self.project_b_skills: list[str] = []
        self.last_exception: Exception | None = None
        self.last_response: dict[str, Any] | None = None
        self.call_completed: bool = False

    def reset(self) -> None:
        """Reset for next test."""
        self.skills_api.reset()
        self.agent = None
        self.last_conversation = None
        self.current_project = None
        self.conversations.clear()
        self.last_exception = None
        self.last_response = None
        self.call_completed = False

    async def start_conversation(
        self, repository: str | None = None, project: str | None = None
    ) -> dict[str, Any]:
        """Start a conversation.

        Args:
            repository: Repository name (optional)
            project: Project name (optional)

        Returns:
            Conversation dict
        """
        logger.info(f'Starting conversation (repo={repository}, project={project})')

        # Create conversation via app-server
        payload = {'title': f'Test conversation - {repository or project or "default"}'}

        try:
            response = await self.http_client.post(
                f'{TEST_APP_SERVER_URL}/app-conversations',
                json=payload,
            )
            self.last_conversation = response.json()
            logger.info(f'Conversation started: {self.last_conversation}')
            return self.last_conversation
        except Exception as e:
            logger.error(f'Failed to start conversation: {e}')
            raise

    async def start_concurrent_conversations(self) -> None:
        """Start multiple conversations concurrently."""
        logger.info('Starting concurrent conversations')
        import asyncio

        tasks = [
            self.start_conversation(project='project_a'),
            self.start_conversation(project='project_b'),
        ]
        await asyncio.gather(*tasks)

    async def initialize_agent(self) -> None:
        """Initialize agent with current skills."""
        logger.info('Initializing agent')
        # In real scenario, this calls app-server which loads skills
        # For testing, we verify the agent would be created with skills
        self.agent = {'skills': await self.get_agent_skills()}

    def _simulate_start_conversation_with_skills(
        self,
        project_name: str | None = None,
        load_public: bool = False,
        load_user: bool = False,
        load_project: bool = True,
        load_org: bool = False,
    ) -> None:
        """Simulate starting a conversation and loading skills.

        This method:
        1. Checks if skills loading is enabled via env var
        2. If disabled, skips calling the agent-server skills API
        3. If enabled, calls the mock API to load skills
        4. Stores results in last_response and last_exception for assertions

        Args:
            project_name: Name of the project (saved to conversations dict)
            load_public: Whether to load global skills
            load_user: Whether to load user skills
            load_project: Whether to load project skills
            load_org: Whether to load organization skills
        """
        # Check if skills loading is enabled via env var
        env_value = os.getenv('OPENHANDS_SKILLS_ENABLED', 'true').lower().strip()
        skills_enabled = env_value in ('true', '1')

        if not skills_enabled:
            logger.info('Skills disabled via OPENHANDS_SKILLS_ENABLED env var')
            self.last_exception = None
            self.last_response = {'skills': [], 'sources': {}}
            self.call_completed = True
            return

        # Skills are enabled; call the mock API
        try:
            payload = {
                'load_public': load_public,
                'load_user': load_user,
                'load_project': load_project,
                'load_org': load_org,
                'project_dir': project_name or 'test_project',
            }
            response = self.skills_api.handle_request_sync(payload)
            self.last_response = response
            self.last_exception = None
            self.call_completed = True
            logger.info(
                f'Skills API call succeeded: {len(response.get("skills", []))} skills loaded'
            )
        except Exception as e:
            logger.warning(f'Skills API call failed: {e}')
            # Graceful degradation: handle error but don't propagate exception
            # This matches real behavior where skills loading errors don't crash the agent
            self.last_exception = None  # No exception raised to caller
            self.last_response = {'skills': [], 'sources': {}}
            self.call_completed = True

    async def get_agent_skills(self) -> list[Any]:
        """Get agent's current skills.

        Returns:
            List of Skill objects or dicts
        """
        # If skills are disabled via env var, return empty
        if os.getenv('OPENHANDS_SKILLS_ENABLED', 'true').lower() not in ('true', '1'):
            logger.info('Skills disabled, returning empty list')
            return []

        # If API was not called (skills disabled), return empty
        if self.skills_api.get_call_count() == 0:
            logger.info('Agent-server not called, returning empty list')
            return []

        # Otherwise, return skills that would have been loaded
        # This is based on the last request's load flags
        last_call = self.skills_api.get_last_call()
        if last_call is None:
            return []

        skills: list[Any] = []

        if last_call.get('load_public'):
            skills.extend(self.skills_api.global_skills)
        if last_call.get('load_user'):
            skills.extend(self.skills_api.user_skills)
        if last_call.get('load_project'):
            skills.extend(self.skills_api.project_skills)
        if last_call.get('load_org'):
            skills.extend(self.skills_api.org_skills)

        logger.info(f'Returning {len(skills)} agent skills')
        return skills


@pytest.fixture
def mock_agent_server_skills() -> Generator[Any, None, None]:
    """Provide mock agent-server /api/skills endpoint.

    Returns:
        MockAgentServerSkillsAPI instance
    """
    from tests.bdd.mocks.agent_server_skills_mock import MockAgentServerSkillsAPI

    api = MockAgentServerSkillsAPI()
    logger.debug('Created mock agent-server skills API')
    yield api
    logger.debug(f'Mock skills API stats: {api}')
    api.reset()


@pytest.fixture
def skills_test_environment(
    mock_agent_server_skills: Any,
) -> Generator[SkillsTestEnvironment, None, None]:
    """Provide complete skills testing environment.

    Args:
        mock_agent_server_skills: Mock skills API

    Returns:
        SkillsTestEnvironment instance
    """
    env = SkillsTestEnvironment(mock_agent_server_skills)
    logger.debug('Created skills test environment')
    yield env
    logger.debug('Cleaning up skills test environment')
    env.reset()
