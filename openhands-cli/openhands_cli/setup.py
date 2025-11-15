import uuid

from openhands.sdk.conversation import visualizer
from openhands.sdk.security.llm_analyzer import LLMSecurityAnalyzer
from prompt_toolkit import HTML, print_formatted_text

from openhands.sdk import Agent, BaseConversation, Conversation, Workspace
from openhands_cli.locations import CONVERSATIONS_DIR, WORK_DIR
from openhands_cli.tui.settings.store import AgentStore
from openhands.sdk.security.confirmation_policy import (
    AlwaysConfirm,
)
from openhands_cli.tui.settings.settings_screen import SettingsScreen
from openhands_cli.tui.visualizer import CLIVisualizer

# register tools
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool


class MissingAgentSpec(Exception):
    """Raised when agent specification is not found or invalid."""

    pass



def load_agent_specs(
    conversation_id: str | None = None,
    *,
    load_user_skills: bool = True,
) -> Agent:
    agent_store = AgentStore()
    agent = agent_store.load(session_id=conversation_id, load_user_skills=load_user_skills)
    if not agent:
        raise MissingAgentSpec(
            'Agent specification not found. Please configure your agent settings.'
        )
    return agent


def verify_agent_exists_or_setup_agent(*, load_user_skills: bool = True) -> Agent:
    """Verify agent specs exists by attempting to load it.

    """
    settings_screen = SettingsScreen()
    try:
        # Keep backward compatibility with upstream tests by not passing kwargs here
        agent = load_agent_specs()
        return agent
    except MissingAgentSpec:
        # For first-time users, show the full settings flow with choice between basic/advanced
        settings_screen.configure_settings(first_time=True)


    # Try once again after settings setup attempt
    # Keep backward compatibility with upstream tests by not passing kwargs here
    return load_agent_specs()


def setup_conversation(
    conversation_id: uuid,
    include_security_analyzer: bool = True,
    *,
    load_user_skills: bool = True,
) -> BaseConversation:
    """
    Setup the conversation with agent.

    Args:
        conversation_id: conversation ID to use. If not provided, a random UUID will be generated.

    Raises:
        MissingAgentSpec: If agent specification is not found or invalid.
    """

    print_formatted_text(
        HTML(f'<white>Initializing agent...</white>')
    )

    agent = load_agent_specs(str(conversation_id), load_user_skills=load_user_skills)



    # Create conversation - agent context is now set in AgentStore.load()
    conversation: BaseConversation = Conversation(
        agent=agent,
        workspace=Workspace(working_dir=WORK_DIR),
        # Conversation will add /<conversation_id> to this path
        persistence_dir=CONVERSATIONS_DIR,
        conversation_id=conversation_id,
        visualizer=CLIVisualizer
    )

    # Security analyzer is set though conversation API now
    if not include_security_analyzer:
        conversation.set_security_analyzer(None)
    else:
        conversation.set_security_analyzer(LLMSecurityAnalyzer())
        conversation.set_confirmation_policy(AlwaysConfirm())

    print_formatted_text(
        HTML(f'<green>✓ Agent initialized with model: {agent.llm.model}</green>')
    )
    return conversation

