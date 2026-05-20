# nue BDD Test Suite

Behavior-driven testing infrastructure for nue using pytest-bdd, Playwright, and mocked services.

## Overview

The BDD test suite enables deterministic, fast testing of nue's agent behavior and frontend functionality without external dependencies.

### Architecture

- **Mocked LLM** (`mocks/llm_mock.py`): Deterministic responses with pattern matching and error injection
- **Mock Sandbox** (`mocks/sandbox_mock.py`): In-memory filesystem simulation for agent testing
- **Playwright E2E** (`conftest.py`): Browser automation for frontend testing
- **HTTP Client** (`utils/api_client.py`): Convenient app-server API calls
- **Step Implementations** (`steps/`): Gherkin step functions
- **Feature Files** (`features/`): BDD scenarios

### Key Features

✓ **Fast** - No real LLM calls, no Docker sandbox startup (typical test: <500ms)
✓ **Deterministic** - Same input always produces same output
✓ **Isolated** - Each test runs with clean mocks and filesystem
✓ **Extensible** - Easy to add new scenarios and customize mocks
✓ **Documented** - Gherkin scenarios serve as living documentation

## Running Tests

### Prerequisites

```bash
poetry install --with test
```

### Run All BDD Tests

```bash
poetry run pytest tests/bdd --gherkin-terminal-reporter
```

Or using Makefile:

```bash
make test-bdd
```

### Run Fast Tests Only (Recommended for Development)

```bash
poetry run pytest tests/bdd -m fast --gherkin-terminal-reporter
```

```bash
make test-bdd-fast
```

### Run Tests with Browser Visible (for Debugging)

```bash
poetry run pytest tests/bdd --headed --gherkin-terminal-reporter
```

```bash
make test-bdd-headed
```

### Run Specific Feature File

```bash
poetry run pytest tests/bdd/features/agent/agent_execution.feature --gherkin-terminal-reporter
```

### Run Tests Matching Pattern

```bash
poetry run pytest tests/bdd -k "chat" --gherkin-terminal-reporter
```

### Watch Mode (Re-run on File Change)

```bash
poetry run ptw tests/bdd
```

## Test Structure

### Directory Layout

```
tests/bdd/
├── conftest.py                          # Main pytest fixtures
├── mocks/
│   ├── llm_mock.py                      # Mock LLM service
│   ├── sandbox_mock.py                  # Mock sandbox + filesystem
│   └── api_server_integration.py         # HTTP mocking
├── steps/
│   ├── agent_steps.py                   # Agent behavior steps
│   ├── frontend_steps.py                 # Frontend interaction steps
│   └── common_steps.py                   # Shared steps
├── features/
│   ├── agent/
│   │   ├── agent_execution.feature
│   │   ├── tool_invocation.feature
│   │   └── sandbox_management.feature
│   ├── frontend/
│   │   ├── chat_interface.feature
│   │   ├── settings.feature
│   │   └── navigation.feature
│   └── integration/
│       ├── end_to_end.feature
│       └── error_handling.feature
├── utils/
│   ├── browser_helpers.py               # Playwright utilities
│   ├── api_client.py                    # HTTP client
│   └── test_data.py                     # Constants and seed data
└── README.md                            # This file
```

### Feature File Format

Feature files use Gherkin syntax:

```gherkin
Feature: Description of behavior
  Context about why this matters

  Scenario: Specific behavior to test
    Given some initial state
    When an action occurs
    Then an outcome is verified
    And another condition holds
```

### Step Implementation Example

```python
from pytest_bdd import given, when, then
from tests.bdd.mocks.llm_mock import LLMMock

@given("an agent session is running")
async def agent_session_running(mock_llm: LLMMock):
    """Initialize an agent session."""
    # Setup code here
    pass

@when("the user sends <message>")
def user_sends_message(message: str):
    """Capture user message."""
    # Action code here
    pass

@then("the LLM is called")
async def llm_called(mock_llm: LLMMock):
    """Verify LLM interaction."""
    # Assertion code here
    pass
```

## Mock Services

### Mock LLM (`mocks/llm_mock.py`)

Deterministic language model with configurable responses:

```python
@pytest.fixture
def mock_llm():
    return LLMMock()

async def test_agent_list_files(mock_llm):
    # Configure response for specific trigger
    mock_llm.configure_response(
        trigger="list files",
        action="run",
        command="find . -type f"
    )

    # Call mock LLM
    response = await mock_llm.call("list files in src/")

    assert response["action"] == "run"
```

#### Error Injection

Simulate LLM failures:

```python
# Raise timeout error once, then succeed
mock_llm.raise_error("timeout", count=1)

with pytest.raises(TimeoutError):
    await mock_llm.call("test")

# Succeeds on second call
response = await mock_llm.call("test")
```

#### Supported Error Types

- `"timeout"` → `TimeoutError`
- `"api_error"` → `RuntimeError`
- `"invalid_response"` → `ValueError`

### Mock Sandbox (`mocks/sandbox_mock.py`)

In-memory filesystem simulation:

```python
@pytest.fixture
def mock_sandbox():
    return MockSandbox()

async def test_sandbox_execution(mock_sandbox):
    # Write file
    mock_sandbox.write_file("test.py", "print('hello')")

    # Execute command
    result = await mock_sandbox.execute("cat test.py")

    assert result.success
    assert "hello" in result.stdout
```

#### Supported Commands

- `cat <file>` - Read file contents
- `find <dir>` - List files
- `ls <dir>` - List directory
- `echo <text>` - Output text
- `pwd` - Current directory
- `mkdir <dir>` - Create directory
- `rm <file>` - Delete file

### API Client (`utils/api_client.py`)

Convenient HTTP client for app-server:

```python
async def test_conversation_flow(http_client):
    async with AppServerAPIClient() as client:
        # Start conversation
        response = await client.start_conversation(title="Test")
        conv_id = response["conversation_id"]

        # Send message
        result = await client.send_message(conv_id, "list files")

        # Get conversation
        conv = await client.get_conversation(conv_id)
        assert len(conv["messages"]) > 0
```

## Fixtures

### Available Fixtures

```python
# Mock services
mock_llm: LLMMock                    # Mock language model
mock_sandbox: MockSandbox            # Mock sandbox with filesystem

# HTTP client
http_client: AppServerAPIClient      # App-server API client

# Browser
browser: Browser                     # Playwright browser instance

# Configuration
bdd_config: dict                     # Test configuration

# Logging
test_logger: Logger                  # Logger for current test

# Utilities
temp_workspace: Path                 # Temporary directory
```

### Fixture Scopes

- **function** (default): New instance for each test
- **session**: Single instance for all tests
- **module**: Single instance for all tests in a module

### Example: Custom Fixture

```python
@pytest.fixture
def configured_llm(mock_llm):
    """LLM pre-configured with common responses."""
    mock_llm.configure_response("list", action="run", command="find .")
    return mock_llm
```

## Test Configuration

### pytest.ini Settings

Located in `pyproject.toml` `[tool.pytest.ini_options]`:

```ini
testpaths = ["tests/bdd"]
asyncio_mode = "auto"
markers = [
    "fast: fast tests with mocked services",
    "slow: slow tests with real services",
    "regression: regression tests",
]
bdd_strict = true  # Fail on undefined steps
```

### Environment Variables

```bash
# Run with verbose output
pytest tests/bdd -vv

# Run with print statements visible
pytest tests/bdd -s

# Run with coverage
pytest tests/bdd --cov=openhands --cov-report=html
```

## Browser Helpers

Playwright utility functions in `utils/browser_helpers.py`:

```python
from tests.bdd.utils.browser_helpers import *

# Element interaction
await wait_for_element(page, ".chat-input", timeout=5)
await fill_input(page, ".chat-input", "hello")
await click_button(page, "Send")
await click_element(page, ".send-button")

# Text assertions
await wait_for_text(page, "Message received")
await wait_for_text_gone(page, "Loading...")

# Element queries
text = await get_text(page, ".response")
value = await get_input_value(page, ".chat-input")
visible = await is_visible(page, ".error")
hidden = await is_hidden(page, ".spinner")

# Navigation
await scroll_to_bottom(page)
await reload_page(page)
await go_back(page)
```

## Adding New Test Scenarios

### Step 1: Create Feature File

Create `.feature` file in appropriate directory:

```gherkin
# tests/bdd/features/agent/my_feature.feature
Feature: My new feature

  Scenario: Specific behavior
    Given some state
    When an action
    Then verify outcome
```

### Step 2: Implement Steps

Add step functions to appropriate `steps/*.py` file:

```python
# tests/bdd/steps/agent_steps.py
from pytest_bdd import given, when, then

@given("some state")
async def setup_state(mock_llm):
    # Implementation
    pass

@when("an action")
async def perform_action(mock_llm):
    # Implementation
    pass

@then("verify outcome")
async def verify_outcome(mock_llm):
    # Implementation
    pass
```

### Step 3: Run Tests

```bash
poetry run pytest tests/bdd/features/agent/my_feature.feature --gherkin-terminal-reporter
```

## Troubleshooting

### Tests Hang or Timeout

Check logs: `logs/bdd-tests.log`

```bash
tail -f logs/bdd-tests.log
```

### Browser Tests Fail

Run with `--headed` to see browser:

```bash
pytest tests/bdd/features/frontend/ --headed
```

Check Playwright browser caching:

```bash
rm -rf ~/.cache/ms-playwright
PLAYWRIGHT_BROWSERS_PATH=$HOME/.cache/playwright poetry run playwright install chromium
```

### Mock LLM Not Configured

Ensure fixture is passed to test:

```python
async def test_something(mock_llm):  # <-- mock_llm fixture
    response = await mock_llm.call("test")
```

### Undefined Step Errors

Implement step in appropriate `steps/*.py` file. Run tests with `-vv` for details:

```bash
pytest tests/bdd -vv
```

## Logging

Logs are written to:
- **Console**: INFO level (normal operation)
- **File**: `logs/bdd-tests.log` (DEBUG level, all details)

View logs:

```bash
# Real-time monitoring
tail -f logs/bdd-tests.log

# Search for errors
grep ERROR logs/bdd-tests.log

# View specific test
grep "test_name" logs/bdd-tests.log
```

## CI Integration

BDD tests run in CI on every PR:

```bash
# Run in CI
make test-bdd

# With coverage
pytest tests/bdd --cov=openhands --cov-report=term-missing
```

## Performance Tips

1. **Use `mock_llm.reset()` after each test** - Already automatic via fixture
2. **Avoid real filesystem operations** - Use `mock_sandbox` instead
3. **Group related tests** - Run specific features: `pytest tests/bdd/features/agent/`
4. **Use `-m fast`** - Run only fast tests: `pytest tests/bdd -m fast`

## Further Reading

- [pytest-bdd Documentation](https://pytest-bdd.readthedocs.io/)
- [Playwright Documentation](https://playwright.dev/python/)
- [Gherkin Syntax Guide](https://cucumber.io/docs/gherkin/reference/)

## Contributing

When adding new BDD tests:

1. Write feature file (Gherkin scenarios)
2. Implement step functions
3. Run tests: `make test-bdd`
4. Check logs if failures: `tail -f logs/bdd-tests.log`
5. Commit with message: "Add BDD tests for [feature]"

## Questions?

Refer to:
- Existing feature files: `tests/bdd/features/`
- Mock implementations: `tests/bdd/mocks/`
- Fixture definitions: `tests/bdd/conftest.py`
- Step examples: `tests/bdd/steps/`
