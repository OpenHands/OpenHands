import os
import shutil

import pytest

from openhands.controller.state.control_flags import IterationControlFlag
from openhands.controller.state.state import State
from openhands.core.message import Message, TextContent
from openhands.events.observation.agent import MicroagentKnowledge
from openhands.microagent import BaseMicroagent
from openhands.utils.prompt import (
    ConversationInstructions,
    PromptManager,
    RepositoryInfo,
    RuntimeInfo,
)


@pytest.fixture
def prompt_dir(tmp_path):
    # Copy contents from "openhands/agenthub/codeact_agent" to the temp directory
    shutil.copytree(
        'openhands/agenthub/codeact_agent/prompts', tmp_path, dirs_exist_ok=True
    )

    # Return the temporary directory path
    return tmp_path


def test_prompt_manager_template_rendering(prompt_dir):
    """Test PromptManager's template rendering functionality."""
    # Create temporary template files
    with open(os.path.join(prompt_dir, 'system_prompt.j2'), 'w') as f:
        f.write("""System prompt: bar""")
    with open(os.path.join(prompt_dir, 'user_prompt.j2'), 'w') as f:
        f.write('User prompt: foo')
    with open(os.path.join(prompt_dir, 'additional_info.j2'), 'w') as f:
        f.write("""
{% if repository_info %}
<REPOSITORY_INFO>
At the user's request, repository {{ repository_info.repo_name }} has been cloned to the current working directory {{ repository_info.repo_directory }}.
</REPOSITORY_INFO>
{% endif %}
""")

    # Test without GitHub repo
    manager = PromptManager(prompt_dir)
    assert manager.get_system_message() == 'System prompt: bar'
    assert manager.get_example_user_message() == 'User prompt: foo'

    # Test with GitHub repo
    manager = PromptManager(prompt_dir=prompt_dir)
    repo_info = RepositoryInfo(
        repo_name='owner/repo', repo_directory='/workspace/repo', branch_name='main'
    )

    # verify its parts are rendered
    system_msg = manager.get_system_message()
    assert 'System prompt: bar' in system_msg

    # Test building additional info
    additional_info = manager.build_workspace_context(
        repository_info=repo_info,
        runtime_info=None,
        repo_instructions='',
        conversation_instructions=None,
    )
    assert '<REPOSITORY_INFO>' in additional_info
    assert (
        "At the user's request, repository owner/repo has been cloned to the current working directory /workspace/repo."
        in additional_info
    )
    assert '</REPOSITORY_INFO>' in additional_info
    assert manager.get_example_user_message() == 'User prompt: foo'

    # Clean up temporary files
    os.remove(os.path.join(prompt_dir, 'system_prompt.j2'))
    os.remove(os.path.join(prompt_dir, 'user_prompt.j2'))
    os.remove(os.path.join(prompt_dir, 'additional_info.j2'))


def test_prompt_manager_file_not_found(prompt_dir):
    """Test PromptManager behavior when a template file is not found."""
    # Test with a non-existent template
    with pytest.raises(FileNotFoundError):
        BaseMicroagent.load(
            os.path.join(prompt_dir, 'micro', 'non_existent_microagent.md')
        )


def test_build_microagent_info(prompt_dir):
    """Test the build_microagent_info method with the microagent_info.j2 template."""
    # Prepare a microagent_info.j2 template file if it doesn't exist
    template_path = os.path.join(prompt_dir, 'microagent_info.j2')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""{% for agent_info in triggered_agents %}
<EXTRA_INFO>
The following information has been included based on a keyword match for "{{ agent_info.trigger }}".
It may or may not be relevant to the user's request.

{{ agent_info.content }}
</EXTRA_INFO>
{% endfor %}
""")

    # Initialize the PromptManager
    manager = PromptManager(prompt_dir=prompt_dir)

    # Test with a single triggered agent
    triggered_agents = [
        MicroagentKnowledge(
            name='test_agent1',
            trigger='keyword1',
            content='This is information from agent 1',
        )
    ]
    result = manager.build_microagent_info(triggered_agents)
    expected = """<EXTRA_INFO>
The following information has been included based on a keyword match for "keyword1".
It may or may not be relevant to the user's request.

This is information from agent 1
</EXTRA_INFO>"""
    assert result.strip() == expected.strip()

    # Test with multiple triggered agents
    triggered_agents = [
        MicroagentKnowledge(
            name='test_agent1',
            trigger='keyword1',
            content='This is information from agent 1',
        ),
        MicroagentKnowledge(
            name='test_agent2',
            trigger='keyword2',
            content='This is information from agent 2',
        ),
    ]
    result = manager.build_microagent_info(triggered_agents)
    expected = """<EXTRA_INFO>
The following information has been included based on a keyword match for "keyword1".
It may or may not be relevant to the user's request.

This is information from agent 1
</EXTRA_INFO>

<EXTRA_INFO>
The following information has been included based on a keyword match for "keyword2".
It may or may not be relevant to the user's request.

This is information from agent 2
</EXTRA_INFO>"""
    assert result.strip() == expected.strip()

    # Test with no triggered agents
    result = manager.build_microagent_info([])
    assert result.strip() == ''


def test_add_turns_left_reminder(prompt_dir):
    """Test adding turns left reminder to messages."""
    # Initialize the PromptManager
    manager = PromptManager(prompt_dir=prompt_dir)

    # Create a State object with specific iteration values
    state = State(
        iteration_flag=IterationControlFlag(
            current_value=3, max_value=10, limit_increase_amount=10
        )
    )

    # Create a list of messages with a user message
    user_message = Message(role='user', content=[TextContent(text='User content')])
    assistant_message = Message(
        role='assistant', content=[TextContent(text='Assistant content')]
    )
    messages = [assistant_message, user_message]

    # Add turns left reminder
    manager.add_turns_left_reminder(messages, state)

    # Check that the reminder was added to the latest user message
    assert len(user_message.content) == 2
    assert (
        'ENVIRONMENT REMINDER: You have 7 turns left to complete the task.'
        in user_message.content[1].text
    )


def test_build_workspace_context_with_repo_and_runtime(prompt_dir):
    """Test building additional info with repository and runtime information."""
    # Create an additional_info.j2 template file
    with open(os.path.join(prompt_dir, 'additional_info.j2'), 'w') as f:
        f.write("""
{% if repository_info %}
<REPOSITORY_INFO>
At the user's request, repository {{ repository_info.repo_name }} has been cloned to directory {{ repository_info.repo_directory }}.
</REPOSITORY_INFO>
{% endif %}

{% if repository_instructions %}
<REPOSITORY_INSTRUCTIONS>
{{ repository_instructions }}
</REPOSITORY_INSTRUCTIONS>
{% endif %}

{% if runtime_info and (runtime_info.available_hosts or runtime_info.additional_agent_instructions) -%}
<RUNTIME_INFORMATION>
{% if runtime_info.available_hosts %}
The user has access to the following hosts for accessing a web application,
each of which has a corresponding port:
{% for host, port in runtime_info.available_hosts.items() %}
* {{ host }} (port {{ port }})
{% endfor %}
{% endif %}

{% if runtime_info.additional_agent_instructions %}
{{ runtime_info.additional_agent_instructions }}
{% endif %}

Today's date is {{ runtime_info.date }}
</RUNTIME_INFORMATION>
{% if conversation_instructions.content %}
<CONVERSATION_INSTRUCTIONS>
{{ conversation_instructions.content }}
</CONVERSATION_INSTRUCTIONS>
{% endif %}
{% endif %}
""")

    # Initialize the PromptManager
    manager = PromptManager(prompt_dir=prompt_dir)

    # Create repository and runtime information
    repo_info = RepositoryInfo(
        repo_name='owner/repo', repo_directory='/workspace/repo', branch_name='main'
    )
    runtime_info = RuntimeInfo(
        date='02/12/1232',
        available_hosts={'example.com': 8080},
        additional_agent_instructions='You know everything about this runtime.',
        working_dir='/workspace',
    )
    repo_instructions = 'This repository contains important code.'

    conversation_instructions = ConversationInstructions(content='additional context')

    # Build additional info
    result = manager.build_workspace_context(
        repository_info=repo_info,
        runtime_info=runtime_info,
        repo_instructions=repo_instructions,
        conversation_instructions=conversation_instructions,
    )

    # Check that all information is included
    assert '<REPOSITORY_INFO>' in result
    assert 'owner/repo' in result
    assert '/workspace/repo' in result
    assert '<REPOSITORY_INSTRUCTIONS>' in result
    assert 'This repository contains important code.' in result
    assert '<RUNTIME_INFORMATION>' in result
    assert 'example.com (port 8080)' in result
    assert 'You know everything about this runtime.' in result
    assert "Today's date is 02/12/1232" in result
    assert 'additional context' in result

    # Clean up
    os.remove(os.path.join(prompt_dir, 'additional_info.j2'))


def test_prompt_manager_initialization_error():
    """Test that PromptManager raises an error if the prompt directory is not set."""
    with pytest.raises(ValueError, match='Prompt directory is not set'):
        PromptManager(None)


def test_prompt_manager_custom_system_prompt_filename(prompt_dir):
    """Test that PromptManager can use a custom system prompt filename."""
    # Create a custom system prompt file
    with open(os.path.join(prompt_dir, 'custom_system.j2'), 'w') as f:
        f.write('Custom system prompt: {{ custom_var }}')

    # Create default system prompt
    with open(os.path.join(prompt_dir, 'system_prompt.j2'), 'w') as f:
        f.write('Default system prompt')

    # Test with custom system prompt filename
    manager = PromptManager(
        prompt_dir=prompt_dir, system_prompt_filename='custom_system.j2'
    )
    system_msg = manager.get_system_message()
    assert 'Custom system prompt:' in system_msg

    # Test without custom system prompt filename (should use default)
    manager_default = PromptManager(prompt_dir=prompt_dir)
    default_msg = manager_default.get_system_message()
    assert 'Default system prompt' in default_msg

    # Clean up
    os.remove(os.path.join(prompt_dir, 'custom_system.j2'))
    os.remove(os.path.join(prompt_dir, 'system_prompt.j2'))


def test_prompt_manager_custom_system_prompt_filename_not_found(prompt_dir):
    """Test that PromptManager raises an error if custom system prompt file is not found."""
    with pytest.raises(
        FileNotFoundError,
        match=r'Prompt file .*/non_existent\.j2 not found',
    ):
        PromptManager(prompt_dir=prompt_dir, system_prompt_filename='non_existent.j2')


def test_jinja2_template_inheritance(prompt_dir):
    """Test that PromptManager._load_template works with Jinja2 template inclusion.

    This test demonstrates that we can use {% include %} to import a base system_prompt.j2
    into other templates without defining any blocks in the base template, and that
    PromptManager._load_template can load these templates correctly.
    """
    # Create base system prompt template (no blocks defined here)
    with open(os.path.join(prompt_dir, 'system_prompt.j2'), 'w') as f:
        f.write("""You are Build.One agent, a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>
Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively.
</ROLE>
""")

    # Create interactive system prompt that imports the base template and adds content
    with open(os.path.join(prompt_dir, 'system_prompt_interactive.j2'), 'w') as f:
        f.write("""{% include "system_prompt.j2" %}

<INTERACTION_RULES>
1. Always respond in a friendly, helpful manner
2. Ask clarifying questions when needed
3. Provide step-by-step explanations
</INTERACTION_RULES>
""")

    # Create long horizon system prompt that imports the base template and adds content
    with open(os.path.join(prompt_dir, 'system_prompt_long_horizon.j2'), 'w') as f:
        f.write("""{% include "system_prompt.j2" %}

<TASK_MANAGEMENT>
1. Break down complex tasks into smaller steps
2. Track progress through a TODO list
3. Focus on one task at a time
</TASK_MANAGEMENT>
""")

    # Test PromptManager._load_template with base system prompt
    base_manager = PromptManager(prompt_dir=prompt_dir)
    base_template = base_manager._load_template('system_prompt.j2')
    base_msg = base_template.render().strip()
    assert 'You are Build.One agent' in base_msg
    assert '<ROLE>' in base_msg
    assert '<INTERACTION_RULES>' not in base_msg
    assert '<TASK_MANAGEMENT>' not in base_msg

    # Test PromptManager._load_template with interactive system prompt
    interactive_manager = PromptManager(
        prompt_dir=prompt_dir, system_prompt_filename='system_prompt_interactive.j2'
    )
    interactive_template = interactive_manager._load_template(
        'system_prompt_interactive.j2'
    )
    interactive_msg = interactive_template.render().strip()
    assert 'You are Build.One agent' in interactive_msg
    assert '<ROLE>' in interactive_msg
    assert '<INTERACTION_RULES>' in interactive_msg
    assert 'Ask clarifying questions when needed' in interactive_msg
    assert '<TASK_MANAGEMENT>' not in interactive_msg

    # Test PromptManager._load_template with long horizon system prompt
    long_horizon_manager = PromptManager(
        prompt_dir=prompt_dir, system_prompt_filename='system_prompt_long_horizon.j2'
    )
    long_horizon_template = long_horizon_manager._load_template(
        'system_prompt_long_horizon.j2'
    )
    long_horizon_msg = long_horizon_template.render().strip()
    assert 'You are Build.One agent' in long_horizon_msg
    assert '<ROLE>' in long_horizon_msg
    assert '<INTERACTION_RULES>' not in long_horizon_msg
    assert '<TASK_MANAGEMENT>' in long_horizon_msg
    assert 'Track progress through a TODO list' in long_horizon_msg

    # Clean up
    os.remove(os.path.join(prompt_dir, 'system_prompt.j2'))
    os.remove(os.path.join(prompt_dir, 'system_prompt_interactive.j2'))
    os.remove(os.path.join(prompt_dir, 'system_prompt_long_horizon.j2'))


def test_prompt_manager_cli_mode_context(prompt_dir):
    """Test that PromptManager.get_system_message() supports cli_mode context parameter."""
    # Create a system prompt template that uses cli_mode conditional
    with open(os.path.join(prompt_dir, 'system_prompt.j2'), 'w') as f:
        f.write("""You are Build.One agent.

{% if cli_mode %}
<CLI_MODE>
You are running in CLI mode. Direct file system access is available.
</CLI_MODE>
{% else %}
<SANDBOX_MODE>
You are running inside sandbox. Container-scoped operations are available.
</SANDBOX_MODE>
{% endif %}

<COMMON_INSTRUCTIONS>
Always be helpful and follow user instructions.
</COMMON_INSTRUCTIONS>""")

    manager = PromptManager(prompt_dir)

    # Test with cli_mode=True
    cli_message = manager.get_system_message(cli_mode=True)
    assert 'You are Build.One agent' in cli_message
    assert '<CLI_MODE>' in cli_message
    assert 'CLI mode' in cli_message
    assert 'Direct file system access' in cli_message
    assert '<SANDBOX_MODE>' not in cli_message
    assert 'inside sandbox' not in cli_message
    assert '<COMMON_INSTRUCTIONS>' in cli_message

    # Test with cli_mode=False
    sandbox_message = manager.get_system_message(cli_mode=False)
    assert 'You are Build.One agent' in sandbox_message
    assert '<SANDBOX_MODE>' in sandbox_message
    assert 'inside sandbox' in sandbox_message
    assert 'Container-scoped operations' in sandbox_message
    assert '<CLI_MODE>' not in sandbox_message
    assert 'CLI mode' not in sandbox_message
    assert '<COMMON_INSTRUCTIONS>' in sandbox_message

    # Test without cli_mode parameter (backward compatibility)
    default_message = manager.get_system_message()
    assert 'You are Build.One agent' in default_message
    assert '<COMMON_INSTRUCTIONS>' in default_message
    # Without cli_mode, the conditional should evaluate to False
    assert '<SANDBOX_MODE>' in default_message
    assert '<CLI_MODE>' not in default_message

    # Test with additional context parameters
    mixed_message = manager.get_system_message(cli_mode=True, custom_var='test_value')
    assert '<CLI_MODE>' in mixed_message
    assert '<COMMON_INSTRUCTIONS>' in mixed_message

    # Verify messages are different based on cli_mode
    assert cli_message != sandbox_message
    assert len(cli_message) != len(sandbox_message)

    # Clean up
    os.remove(os.path.join(prompt_dir, 'system_prompt.j2'))
