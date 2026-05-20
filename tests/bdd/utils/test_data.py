"""Test data and fixtures for BDD tests.

Provides constants, seed data, and sample responses for consistent test scenarios.
"""

from __future__ import annotations

from typing import Any

# User constants
TEST_USER_ID = 'test-user-123'
TEST_USER_EMAIL = 'test@example.com'
TEST_USER_NAME = 'Test User'

# Conversation constants
TEST_CONVERSATION_ID = 'test-conv-001'
TEST_CONVERSATION_TITLE = 'Test Conversation'

# LLM Model constants
DEFAULT_TEST_MODEL = 'gpt-4'
AVAILABLE_TEST_MODELS = [
    'gpt-4',
    'gpt-3.5-turbo',
    'claude-3-opus',
    'claude-3-sonnet',
]

# Port constants
TEST_APP_SERVER_PORT = 9999
TEST_APP_SERVER_URL = f'http://localhost:{TEST_APP_SERVER_PORT}'
TEST_FRONTEND_PORT = 3001
TEST_FRONTEND_URL = f'http://localhost:{TEST_FRONTEND_PORT}'

# Timeout constants (in seconds)
DEFAULT_TIMEOUT = 5.0
LONG_TIMEOUT = 30.0
SHORT_TIMEOUT = 1.0

# Sample LLM responses
SAMPLE_LLM_RESPONSES = {
    'list_files_response': {
        'action': 'run',
        'command': "find . -type f -name '*.py' | head -10",
        'thought': "User wants to see Python files. I'll use find to list them.",
    },
    'edit_file_response': {
        'action': 'edit',
        'file': 'src/main.py',
        'old_content': "def hello():\n    print('old')",
        'new_content': "def hello():\n    print('hello world')",
        'thought': "User wants me to edit the function. I'll update it.",
    },
    'analysis_response': {
        'action': 'think',
        'thought': 'Let me analyze this code carefully.',
        'result': 'This code appears to be a simple Python script. It imports sys and defines a main function.',
    },
    'help_response': {
        'action': 'ask_followup',
        'question': 'Could you provide more context about what you want to achieve?',
        'thought': 'I need more information to help effectively.',
    },
}

# Sample file structures
SAMPLE_PROJECT_STRUCTURE = {
    'files': [
        'README.md',
        'requirements.txt',
        'setup.py',
        'src/main.py',
        'src/config.py',
        'tests/test_main.py',
        'tests/test_config.py',
    ]
}

# Sample code content
SAMPLE_PYTHON_CODE = '''#!/usr/bin/env python3
"""Sample Python module."""

def hello(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"

def main() -> None:
    """Main entry point."""
    print(hello("World"))

if __name__ == "__main__":
    main()
'''

SAMPLE_README_CONTENT = """# Sample Project

This is a sample project for testing purposes.

## Installation

```
pip install -r requirements.txt
```

## Usage

```
python src/main.py
```

## Testing

```
pytest tests/
```
"""

# Sample user settings
SAMPLE_USER_SETTINGS = {
    'id': TEST_USER_ID,
    'name': TEST_USER_NAME,
    'email': TEST_USER_EMAIL,
    'llm_model': DEFAULT_TEST_MODEL,
    'llm_api_key': 'sk-test-123456789',
    'llm_base_url': None,
    'timezone': 'UTC',
    'language': 'en',
}

# Sample conversation object
SAMPLE_CONVERSATION = {
    'id': TEST_CONVERSATION_ID,
    'title': TEST_CONVERSATION_TITLE,
    'created_at': '2024-05-19T21:00:00Z',
    'updated_at': '2024-05-19T21:00:00Z',
    'agent_state': 'AWAITING_USER_INPUT',
    'sandbox_status': 'RUNNING',
    'messages': [],
}

# Sample messages
SAMPLE_USER_MESSAGE = {
    'id': 'msg-001',
    'role': 'user',
    'content': 'List the files in the project',
    'created_at': '2024-05-19T21:00:00Z',
}

SAMPLE_ASSISTANT_MESSAGE = {
    'id': 'msg-002',
    'role': 'assistant',
    'content': "I'll list the files for you.",
    'action': 'run',
    'command': 'find . -type f',
    'created_at': '2024-05-19T21:00:01Z',
}

# Sample MCP servers
SAMPLE_MCP_SERVERS = [
    {
        'id': 'mcp-001',
        'name': 'Filesystem',
        'type': 'filesystem',
        'enabled': True,
    },
    {
        'id': 'mcp-002',
        'name': 'Git',
        'type': 'git',
        'enabled': True,
    },
]

# Feature file paths for reference
FEATURE_PATHS = {
    'agent_execution': 'tests/bdd/features/agent/agent_execution.feature',
    'tool_invocation': 'tests/bdd/features/agent/tool_invocation.feature',
    'sandbox_management': 'tests/bdd/features/agent/sandbox_management.feature',
    'chat_interface': 'tests/bdd/features/frontend/chat_interface.feature',
    'settings': 'tests/bdd/features/frontend/settings.feature',
    'navigation': 'tests/bdd/features/frontend/navigation.feature',
    'end_to_end': 'tests/bdd/features/integration/end_to_end.feature',
    'error_handling': 'tests/bdd/features/integration/error_handling.feature',
}


def get_sample_llm_response(scenario: str) -> dict[str, Any]:
    """Get sample LLM response for a scenario.

    Args:
        scenario: Scenario name (e.g., "list_files_response")

    Returns:
        LLM response dict
    """
    return SAMPLE_LLM_RESPONSES.get(
        scenario,
        {
            'action': 'think',
            'thought': f'Processing {scenario}',
            'result': 'Done',
        },
    )


def get_default_user_settings() -> dict[str, Any]:
    """Get default user settings for tests.

    Returns:
        User settings dict
    """
    return SAMPLE_USER_SETTINGS.copy()


def get_default_conversation() -> dict[str, Any]:
    """Get default conversation object for tests.

    Returns:
        Conversation dict
    """
    return SAMPLE_CONVERSATION.copy()


def get_default_mcp_servers() -> list[dict[str, Any]]:
    """Get default MCP servers for tests.

    Returns:
        List of MCP server dicts
    """
    return [server.copy() for server in SAMPLE_MCP_SERVERS]
