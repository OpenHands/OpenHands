"""Test scenarios for agent execution.

Uses pytest-bdd to convert features/agent/agent_execution.feature to tests.
"""

from pathlib import Path

from pytest_bdd import scenario

# Import steps so pytest-bdd can discover them
# This makes step definitions available in this module's namespace
from tests.bdd.steps.agent_steps import *  # noqa: F401, F403

# Get the path to the feature file
feature_file = Path(__file__).parent / 'agent_execution.feature'


@scenario(str(feature_file), 'User sends a message and agent responds')
def test_user_sends_message_agent_responds():
    """Test user sends message and agent responds."""
    pass
