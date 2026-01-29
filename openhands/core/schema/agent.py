# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from enum import Enum


class AgentState(str, Enum):
    LOADING = 'loading'
    """The agent is loading.
    """

    RUNNING = 'running'
    """The agent is running.
    """

    AWAITING_USER_INPUT = 'awaiting_user_input'
    """The agent is awaiting user input.
    """

    PAUSED = 'paused'
    """The agent is paused.
    """

    STOPPED = 'stopped'
    """The agent is stopped.
    """

    FINISHED = 'finished'
    """The agent is finished with the current task.
    """

    REJECTED = 'rejected'
    """The agent rejects the task.
    """

    ERROR = 'error'
    """An error occurred during the task.
    """

    AWAITING_USER_CONFIRMATION = 'awaiting_user_confirmation'
    """The agent is awaiting user confirmation.
    """

    USER_CONFIRMED = 'user_confirmed'
    """The user confirmed the agent's action.
    """

    USER_REJECTED = 'user_rejected'
    """The user rejected the agent's action.
    """

    RATE_LIMITED = 'rate_limited'
    """The agent is rate limited.
    """
