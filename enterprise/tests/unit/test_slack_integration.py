from unittest.mock import MagicMock

import pytest
from integrations.slack.slack_manager import SlackManager


@pytest.fixture
def slack_manager():
    # Mock the token_manager constructor
    slack_manager = SlackManager(token_manager=MagicMock())
    return slack_manager


@pytest.mark.parametrize(
    'message,expected',
    [
        ('OpenHands/Openhands', 'OpenHands/Openhands'),
        ('deploy repo', 'deploy'),
        ('use hello world', None),
    ],
)
def test_infer_repo_from_message(message, expected, slack_manager):
    # Test the extracted function
    result = slack_manager._infer_repo_from_message(message)
    assert result == expected
