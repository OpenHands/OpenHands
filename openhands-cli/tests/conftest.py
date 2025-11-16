from unittest.mock import patch

import pytest


# Fixture: mock_verified_models - Simplified model data
@pytest.fixture
def mock_verified_models():
    with (
        patch(
            'openhands_cli.user_actions.settings_action.VERIFIED_MODELS',
            {
                'openai': ['gpt-4o', 'gpt-4o-mini'],
                'anthropic': ['claude-3-5-sonnet', 'claude-3-5-haiku'],
            },
        ),
        patch(
            'openhands_cli.user_actions.settings_action.UNVERIFIED_MODELS_EXCLUDING_BEDROCK',
            {
                'openai': ['gpt-custom'],
                'anthropic': [],
                'custom': ['my-model'],
            },
        ),
    ):
        yield


# Fixture: mock_cli_interactions - Reusable CLI mock patterns
@pytest.fixture
def mock_cli_interactions():
    class Mocks:
        def __init__(self):
            self.p_confirm = patch(
                'openhands_cli.user_actions.settings_action.cli_confirm'
            )
            self.p_text = patch(
                'openhands_cli.user_actions.settings_action.cli_text_input'
            )
            self.cli_confirm = None
            self.cli_text_input = None

        def start(self):
            self.cli_confirm = self.p_confirm.start()
            self.cli_text_input = self.p_text.start()
            return self

        def stop(self):
            self.p_confirm.stop()
            self.p_text.stop()

    mocks = Mocks().start()
    try:
        yield mocks
    finally:
        mocks.stop()


# Fixture: skip CLI terminal compatibility guard for non-interactive tests
@pytest.fixture
def skip_terminal_check_env(monkeypatch):
    monkeypatch.setenv('OPENHANDS_CLI_SKIP_TTY_CHECK', '1')
    yield
    monkeypatch.delenv('OPENHANDS_CLI_SKIP_TTY_CHECK', raising=False)
