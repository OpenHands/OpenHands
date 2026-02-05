# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# ============================================
# Agent Exceptions
# ============================================


class AgentError(Exception):
    """Base class for all agent exceptions."""

    pass


class AgentNoInstructionError(AgentError):
    def __init__(self, message: str = 'Instruction must be provided') -> None:
        super().__init__(message)


class AgentEventTypeError(AgentError):
    def __init__(self, message: str = 'Event must be a dictionary') -> None:
        super().__init__(message)


class AgentAlreadyRegisteredError(AgentError):
    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            message = f"Agent class already registered under '{name}'"
        else:
            message = 'Agent class already registered'
        super().__init__(message)


class AgentNotRegisteredError(AgentError):
    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            message = f"No agent class registered under '{name}'"
        else:
            message = 'No agent class registered'
        super().__init__(message)


class AgentStuckInLoopError(AgentError):
    def __init__(self, message: str = 'Agent got stuck in a loop') -> None:
        super().__init__(message)


# ============================================
# Agent Controller Exceptions
# ============================================


class TaskInvalidStateError(Exception):
    def __init__(self, state: str | None = None) -> None:
        if state is not None:
            message = f'Invalid state {state}'
        else:
            message = 'Invalid state'
        super().__init__(message)


# ============================================
# LLM Exceptions
# ============================================


# This exception gets sent back to the LLM
# It might be malformed JSON
class LLMMalformedActionError(Exception):
    def __init__(self, message: str = 'Malformed response') -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


# This exception gets sent back to the LLM
# For some reason, the agent did not return an action
class LLMNoActionError(Exception):
    def __init__(self, message: str = 'Agent must return an action') -> None:
        super().__init__(message)


# This exception gets sent back to the LLM
# The LLM output did not include an action, or the action was not the expected type
class LLMResponseError(Exception):
    def __init__(
        self, message: str = 'Failed to retrieve action from LLM response'
    ) -> None:
        super().__init__(message)


# This exception should be retried
# Typically, after retry with a non-zero temperature, the LLM will return a response
class LLMNoResponseError(Exception):
    def __init__(
        self,
        message: str = 'LLM did not return a response. This is only seen in Gemini models so far.',
    ) -> None:
        super().__init__(message)


class UserCancelledError(Exception):
    def __init__(self, message: str = 'User cancelled the request') -> None:
        super().__init__(message)


class OperationCancelled(Exception):
    """Exception raised when an operation is cancelled (e.g. by a keyboard interrupt)."""

    def __init__(self, message: str = 'Operation was cancelled') -> None:
        super().__init__(message)


class LLMContextWindowExceedError(RuntimeError):
    def __init__(
        self,
        message: str = 'Conversation history longer than LLM context window limit. Consider turning on enable_history_truncation config to avoid this error',
    ) -> None:
        super().__init__(message)


# ============================================
# LLM function calling Exceptions
# ============================================


class FunctionCallConversionError(Exception):
    """Exception raised when FunctionCallingConverter failed to convert a non-function call message to a function call message.

    This typically happens when there's a malformed message (e.g., missing <function=...> tags). But not due to LLM output.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class FunctionCallValidationError(Exception):
    """Exception raised when FunctionCallingConverter failed to validate a function call message.

    This typically happens when the LLM outputs unrecognized function call / parameter names / values.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class FunctionCallNotExistsError(Exception):
    """Exception raised when an LLM call a tool that is not registered."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ============================================
# Agent Runtime Exceptions
# ============================================


class AgentRuntimeError(Exception):
    """Base class for all agent runtime exceptions."""

    pass


class AgentRuntimeBuildError(AgentRuntimeError):
    """Exception raised when an agent runtime build operation fails."""

    pass


class AgentRuntimeTimeoutError(AgentRuntimeError):
    """Exception raised when an agent runtime operation times out."""

    pass


class AgentRuntimeUnavailableError(AgentRuntimeError):
    """Exception raised when an agent runtime is unavailable."""

    pass


class AgentRuntimeNotReadyError(AgentRuntimeUnavailableError):
    """Exception raised when an agent runtime is not ready."""

    pass


class AgentRuntimeDisconnectedError(AgentRuntimeUnavailableError):
    """Exception raised when an agent runtime is disconnected."""

    pass


class AgentRuntimeNotFoundError(AgentRuntimeUnavailableError):
    """Exception raised when an agent runtime is not found."""

    pass


# ============================================
# Browser Exceptions
# ============================================


class BrowserInitException(Exception):
    def __init__(
        self, message: str = 'Failed to initialize browser environment'
    ) -> None:
        super().__init__(message)


class BrowserUnavailableException(Exception):
    def __init__(
        self,
        message: str = 'Browser environment is not available, please check if has been initialized',
    ) -> None:
        super().__init__(message)


# ============================================
# Microagent Exceptions
# ============================================


class MicroagentError(Exception):
    """Base exception for all microagent errors."""

    pass


class MicroagentValidationError(MicroagentError):
    """Raised when there's a validation error in microagent metadata."""

    def __init__(self, message: str = 'Microagent validation failed') -> None:
        super().__init__(message)
