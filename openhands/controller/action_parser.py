# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from abc import ABC, abstractmethod
from typing import Any

from openhands.events.action import Action


class ActionParseError(Exception):
    """Exception raised when the response from the LLM cannot be parsed into an action."""

    def __init__(self, error: str):
        self.error = error

    def __str__(self) -> str:
        return self.error


class ResponseParser(ABC):
    """This abstract base class is a general interface for an response parser dedicated to
    parsing the action from the response from the LLM.
    """

    def __init__(
        self,
    ) -> None:
        # Need pay attention to the item order in self.action_parsers
        self.action_parsers: list[ActionParser] = []

    @abstractmethod
    def parse(self, response: Any) -> Action:
        """Parses the action from the response from the LLM.

        Parameters:
        - response: The response from the LLM, which can be a string or a dictionary.

        Returns:
        - action (Action): The action parsed from the response.
        """
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> str:
        """Parses the action from the response from the LLM.

        Parameters:
        - response: The response from the LLM, which can be a string or a dictionary.

        Returns:
        - action_str (str): The action str parsed from the response.
        """
        pass

    @abstractmethod
    def parse_action(self, action_str: str) -> Action:
        """Parses the action from the response from the LLM.

        Parameters:
        - action_str (str): The response from the LLM.

        Returns:
        - action (Action): The action parsed from the response.
        """
        pass


class ActionParser(ABC):
    """This abstract base class is a general interface for an action parser dedicated to
    parsing the action from the action str from the LLM.
    """

    @abstractmethod
    def check_condition(self, action_str: str) -> bool:
        """Check if the action string can be parsed by this parser."""
        pass

    @abstractmethod
    def parse(self, action_str: str) -> Action:
        """Parses the action from the action string from the LLM response."""
        pass
