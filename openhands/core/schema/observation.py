# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from enum import Enum


class ObservationType(str, Enum):
    READ = 'read'
    """The content of a file
    """

    WRITE = 'write'

    EDIT = 'edit'

    BROWSE = 'browse'
    """The HTML content of a URL
    """

    RUN = 'run'
    """The output of a command
    """

    RUN_IPYTHON = 'run_ipython'
    """Runs a IPython cell.
    """

    CHAT = 'chat'
    """A message from the user
    """

    DELEGATE = 'delegate'
    """The result of a task delegated to another agent
    """

    MESSAGE = 'message'

    ERROR = 'error'

    SUCCESS = 'success'

    NULL = 'null'

    THINK = 'think'

    AGENT_STATE_CHANGED = 'agent_state_changed'

    USER_REJECTED = 'user_rejected'

    CONDENSE = 'condense'
    """Result of a condensation operation."""

    RECALL = 'recall'
    """Result of a recall operation. This can be the workspace context, a microagent, or other types of information."""

    MCP = 'mcp'
    """Result of a MCP Server operation"""

    DOWNLOAD = 'download'
    """Result of downloading/opening a file via the browser"""

    TASK_TRACKING = 'task_tracking'
    """Result of a task tracking operation"""

    LOOP_DETECTION = 'loop_detection'
    """Results of a dead-loop detection"""
