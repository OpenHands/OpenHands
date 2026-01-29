# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import ast
import re

from openhands.controller.action_parser import ActionParser, ResponseParser
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    BrowseInteractiveAction,
)


class BrowsingResponseParser(ResponseParser):
    def __init__(self) -> None:
        # Need to pay attention to the item order in self.action_parsers
        super().__init__()
        self.action_parsers = [BrowsingActionParserMessage()]
        self.default_parser = BrowsingActionParserBrowseInteractive()

    def parse(
        self, response: str | dict[str, list[dict[str, dict[str, str | None]]]]
    ) -> Action:
        if isinstance(response, str):
            action_str = response
        else:
            action_str = self.parse_response(response)
        return self.parse_action(action_str)

    def parse_response(
        self, response: dict[str, list[dict[str, dict[str, str | None]]]]
    ) -> str:
        action_str = response['choices'][0]['message']['content']
        if action_str is None:
            return ''
        action_str = action_str.strip()
        # Ensure action_str ends with ')```'
        if action_str:
            if not action_str.endswith('```'):
                if action_str.endswith(')'):
                    action_str += '```'  # prevent duplicate ending paranthesis, e.g. send_msg_to_user('Done'))
                else:
                    action_str += ')```'  # expected format
        logger.debug(action_str)
        return action_str

    def parse_action(self, action_str: str) -> Action:
        for action_parser in self.action_parsers:
            if action_parser.check_condition(action_str):
                return action_parser.parse(action_str)
        return self.default_parser.parse(action_str)


class BrowsingActionParserMessage(ActionParser):
    """Parser action:
    - BrowseInteractiveAction(browser_actions) - unexpected response format, message back to user
    """

    def __init__(self) -> None:
        pass

    def check_condition(self, action_str: str) -> bool:
        return '```' not in action_str

    def parse(self, action_str: str) -> Action:
        msg = f'send_msg_to_user("""{action_str}""")'
        return BrowseInteractiveAction(
            browser_actions=msg,
            thought=action_str,
            browsergym_send_msg_to_user=action_str,
        )


class BrowsingActionParserBrowseInteractive(ActionParser):
    """Parser action:
    - BrowseInteractiveAction(browser_actions) - handle send message to user function call in BrowserGym
    """

    def __init__(self) -> None:
        pass

    def check_condition(self, action_str: str) -> bool:
        return True

    def parse(self, action_str: str) -> Action:
        # parse the action string into browser_actions and thought
        # the LLM can return only one string, or both

        # when both are returned, it looks like this:
        ### Based on the current state of the page and the goal of finding out the president of the USA, the next action should involve searching for information related to the president.
        ### To achieve this, we can navigate to a reliable source such as a search engine or a specific website that provides information about the current president of the USA.
        ### Here is an example of a valid action to achieve this:
        ### ```
        ### goto('https://www.whitehouse.gov/about-the-white-house/presidents/'
        # in practice, BrowsingResponseParser.parse_response also added )``` to the end of the string

        # when the LLM returns only one string, it looks like this:
        ### goto('https://www.whitehouse.gov/about-the-white-house/presidents/')
        # and parse_response added )``` to the end of the string
        parts = action_str.split('```')
        browser_actions = (
            parts[1].strip() if parts[1].strip() != '' else parts[0].strip()
        )
        thought = parts[0].strip() if parts[1].strip() != '' else ''

        # if the LLM wants to talk to the user, we extract the message
        msg_content = ''
        for sub_action in browser_actions.split('\n'):
            if 'send_msg_to_user(' in sub_action:
                try:
                    tree = ast.parse(sub_action)
                    args = tree.body[0].value.args  # type: ignore
                    msg_content = args[0].value
                except SyntaxError:
                    logger.error(f'Error parsing action: {sub_action}')
                    # the syntax was not correct, but we can still try to get the message
                    # e.g. send_msg_to_user("Hello, world!") or send_msg_to_user('Hello, world!'
                    match = re.search(r'send_msg_to_user\((["\'])(.*?)\1\)', sub_action)
                    if match:
                        msg_content = match.group(2)
                    else:
                        msg_content = ''

        return BrowseInteractiveAction(
            browser_actions=browser_actions,
            thought=thought,
            browsergym_send_msg_to_user=msg_content,
        )
