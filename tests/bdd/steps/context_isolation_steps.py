"""Steps verifying that nue agent contexts are free of OpenHands-specific content."""

from __future__ import annotations

import logging
import os

import pytest
from pytest_bdd import given, parsers, then

import openhands.sdk.agent.base as _sdk_agent_base
from openhands.sdk.context.prompts.prompt import render_template

logger = logging.getLogger(__name__)


class SystemPromptContext:
    """Holds the rendered system prompt for assertion steps."""

    def __init__(self) -> None:
        self.rendered: str = ''


@pytest.fixture
def system_prompt_context() -> SystemPromptContext:
    return SystemPromptContext()


@given('the server builds the system prompt for a default non-planning agent')
def given_default_agent_prompt(system_prompt_context: SystemPromptContext) -> None:
    sdk_prompts_dir = os.path.join(os.path.dirname(_sdk_agent_base.__file__), 'prompts')
    # _apply_server_agent_overrides does NOT set system_prompt_filename for
    # non-planning agents; the SDK default 'system_prompt.j2' is used.
    system_prompt_context.rendered = render_template(
        prompt_dir=sdk_prompts_dir,
        template_name='system_prompt.j2',
        cli_mode=False,
        enable_browser=False,
        security_policy_filename=None,
        model_name='test-model',
    )
    logger.info(
        'Rendered %d chars of system prompt', len(system_prompt_context.rendered)
    )


@then(parsers.parse('the system prompt does not contain "{phrase}"'))
def then_prompt_absent(system_prompt_context: SystemPromptContext, phrase: str) -> None:
    assert phrase not in system_prompt_context.rendered, (
        f'System prompt must not contain "{phrase}".\n'
        f'This OpenHands-specific content is injected by the SDK default '
        f'system_prompt.j2. Fix: override system_prompt_filename in '
        f'_apply_server_agent_overrides for non-planning agents.'
    )
