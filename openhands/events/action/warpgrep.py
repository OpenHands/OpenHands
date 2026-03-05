from dataclasses import dataclass
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class WarpGrepAction(Action):
    query: str
    thought: str = ''
    action: str = ActionType.WARPGREP_SEARCH
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = ActionSecurityRisk.LOW

    @property
    def message(self) -> str:
        return f'Searching the codebase with WarpGrep:\n```\n{self.query}\n```'

    def __str__(self) -> str:
        ret = '**WarpGrepAction**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'QUERY: {self.query}'
        return ret
