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
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

import pytest
from playwright.async_api import Browser, async_playwright

from tests.bdd.mocks.llm_mock import LLMMock
from tests.bdd.mocks.sandbox_mock import MockSandbox
from tests.bdd.utils.api_client import AppServerAPIClient
from tests.bdd.utils.test_data import TEST_APP_SERVER_URL

# Import step modules for pytest-bdd discovery
from tests.bdd.steps import agent_steps  # noqa: F401
from tests.bdd.steps import common_steps  # noqa: F401
from tests.bdd.steps import frontend_steps  # noqa: F401


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
        self.llm: Optional[Any] = None
        self.messages: list[dict[str, Any]] = []
        self.last_response: Optional[dict[str, Any]] = None
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
    logger.debug(f'Agent context stats: messages={len(context.messages)}, calls={context.llm_call_count}')
    context.reset()
