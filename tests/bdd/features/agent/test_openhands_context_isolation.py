"""Test scenarios for OpenHands context isolation.

Verifies that nue agent system prompts are free of OpenHands-specific branding
and instructions injected via the SDK's default system_prompt.j2.
"""

from pathlib import Path

from pytest_bdd import scenario

from tests.bdd.steps.context_isolation_steps import *  # noqa: F401, F403

feature_file = Path(__file__).parent / 'openhands_context_isolation.feature'


@scenario(
    str(feature_file), 'Default agent system prompt has no OpenHands identity claim'
)
def test_no_openhands_identity():
    pass


@scenario(
    str(feature_file), 'Default agent system prompt has no AGENTS.md memory instruction'
)
def test_no_agents_md():
    pass


@scenario(
    str(feature_file), 'Default agent system prompt has no OpenHands commit identity'
)
def test_no_openhands_commit_identity():
    pass


@scenario(
    str(feature_file), 'Default agent system prompt has no OpenHands documentation link'
)
def test_no_openhands_docs_link():
    pass
