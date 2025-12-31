from dataclasses import dataclass
from typing import Any

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk
from openhands.version import get_version


@dataclass
class MessageAction(Action):
    content: str
    file_urls: list[str] | None = None
    image_urls: list[str] | None = None
    wait_for_response: bool = False
    action: str = ActionType.MESSAGE
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        return self.content

    @property
    def images_urls(self) -> list[str] | None:
        # Deprecated alias for backward compatibility
        return self.image_urls

    @images_urls.setter
    def images_urls(self, value: list[str] | None) -> None:
        self.image_urls = value

    def __str__(self) -> str:
        ret = f'**MessageAction** (source={self.source})\n'
        ret += f'CONTENT: {self.content}'
        if self.image_urls:
            for url in self.image_urls:
                ret += f'\nIMAGE_URL: {url}'
        if self.file_urls:
            for url in self.file_urls:
                ret += f'\nFILE_URL: {url}'
        return ret


@dataclass
class SystemMessageAction(Action):
    """Action that represents a system message for an agent, including the system prompt
    and available tools. This should be the first message in the event stream.
    """

    content: str
    tools: list[Any] | None = None
    openhands_version: str | None = get_version()
    agent_class: str | None = None
    action: ActionType = ActionType.SYSTEM

    @property
    def message(self) -> str:
        return self.content

    def __str__(self) -> str:
        ret = f'**SystemMessageAction** (source={self.source})\n'
        ret += f'CONTENT: {self.content}'
        if self.tools:
            ret += f'\nTOOLS: {len(self.tools)} tools available'
        if self.agent_class:
            ret += f'\nAGENT_CLASS: {self.agent_class}'
        return ret
