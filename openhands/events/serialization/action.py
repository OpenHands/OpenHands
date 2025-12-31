from typing import Any

from openhands.core.exceptions import LLMMalformedActionError
from openhands.events.action.action import Action, ActionSecurityRisk
from openhands.events.action.agent import (
    AgentDelegateAction,
    AgentFinishAction,
    AgentRejectAction,
    AgentThinkAction,
    ChangeAgentStateAction,
    CondensationAction,
    CondensationRequestAction,
    LoopRecoveryAction,
    RecallAction,
    TaskTrackingAction,
)
from openhands.events.action.browse import BrowseInteractiveAction, BrowseURLAction
from openhands.events.action.commands import (
    CmdRunAction,
    IPythonRunCellAction,
)
from openhands.events.action.empty import NullAction
from openhands.events.action.files import (
    FileEditAction,
    FileReadAction,
    FileWriteAction,
)
from openhands.events.action.mcp import MCPAction
from openhands.events.action.message import MessageAction, SystemMessageAction

actions = (
    NullAction,
    CmdRunAction,
    IPythonRunCellAction,
    BrowseURLAction,
    BrowseInteractiveAction,
    FileReadAction,
    FileWriteAction,
    FileEditAction,
    AgentThinkAction,
    AgentFinishAction,
    AgentRejectAction,
    AgentDelegateAction,
    RecallAction,
    ChangeAgentStateAction,
    MessageAction,
    SystemMessageAction,
    CondensationAction,
    CondensationRequestAction,
    MCPAction,
    TaskTrackingAction,
    LoopRecoveryAction,
)

ACTION_TYPE_TO_CLASS = {action_class.action: action_class for action_class in actions}  # type: ignore[attr-defined]


def handle_action_deprecated_args(args: dict[str, Any]) -> dict[str, Any]:
    # keep_prompt has been deprecated in https://github.com/OpenHands/OpenHands/pull/4881
    if 'keep_prompt' in args:
        args.pop('keep_prompt')

    # task_completed has been deprecated - remove it from args to maintain backward compatibility
    if 'task_completed' in args:
        args.pop('task_completed')

    # Handle translated_ipython_code deprecation
    if 'translated_ipython_code' in args:
        code = args.pop('translated_ipython_code')

        # Check if it's a file_editor call using a prefix check for efficiency
        file_editor_prefix = 'print(file_editor(**'
        if (
            code is not None
            and code.startswith(file_editor_prefix)
            and code.endswith('))')
        ):
            try:
                # Extract and evaluate the dictionary string
                import ast

                # Extract the dictionary string between the prefix and the closing parentheses
                dict_str = code[len(file_editor_prefix) : -2]  # Remove prefix and '))'
                file_args = ast.literal_eval(dict_str)

                # Update args with the extracted file editor arguments
                args.update(file_args)
            except (ValueError, SyntaxError):
                # If parsing fails, just remove the translated_ipython_code
                pass

        if args.get('command') == 'view':
            args.pop(
                'command'
            )  # "view" will be translated to FileReadAction which doesn't have a command argument

    return args


def action_from_dict(action: dict) -> Action:
    if not isinstance(action, dict):
        raise LLMMalformedActionError('action must be a dictionary')
    action = action.copy()
    if 'action' not in action:
        raise LLMMalformedActionError(f"'action' key is not found in {action=}")
    if not isinstance(action['action'], str):
        raise LLMMalformedActionError(
            f"'{action['action']=}' is not defined. Available actions: {ACTION_TYPE_TO_CLASS.keys()}"
        )
    action_class = ACTION_TYPE_TO_CLASS.get(action['action'])
    if action_class is None:
        raise LLMMalformedActionError(
            f"'{action['action']=}' is not defined. Available actions: {ACTION_TYPE_TO_CLASS.keys()}"
        )
    args = action.get('args', {})
    # Remove timestamp from args if present
    timestamp = args.pop('timestamp', None)

    # compatibility for older event streams
    # is_confirmed has been renamed to confirmation_state
    is_confirmed = args.pop('is_confirmed', None)
    if is_confirmed is not None:
        args['confirmation_state'] = is_confirmed

    # images_urls has been renamed to image_urls
    if 'images_urls' in args:
        args['image_urls'] = args.pop('images_urls')

    # Handle security_risk deserialization
    if 'security_risk' in args and args['security_risk'] is not None:
        try:
            # Convert numeric value (int) back to enum
            args['security_risk'] = ActionSecurityRisk(args['security_risk'])
        except (ValueError, TypeError):
            # If conversion fails, remove the invalid value
            args.pop('security_risk')

    # handle deprecated args
    args = handle_action_deprecated_args(args)

    try:
        decoded_action = action_class(**args)
        if 'timeout' in action:
            blocking = args.get('blocking', False)
            decoded_action.set_hard_timeout(action['timeout'], blocking=blocking)

        # Set timestamp if it was provided
        if timestamp:
            decoded_action._timestamp = timestamp

    except TypeError as e:
        raise LLMMalformedActionError(
            f'action={action} has the wrong arguments: {str(e)}'
        )
    assert isinstance(decoded_action, Action)
    return decoded_action
