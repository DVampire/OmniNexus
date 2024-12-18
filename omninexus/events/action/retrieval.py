from dataclasses import dataclass
from typing import ClassVar

from omninexus.core.schema import ActionType
from omninexus.events.action.action import Action, ActionSecurityRisk


@dataclass
class RelevantResearchRetrievalAction(Action):
    browser_actions: str
    thought: str = ''
    browsergym_send_msg_to_user: str = ''
    action: str = ActionType.BROWSE_INTERACTIVE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return (
            f'I am interacting with the browser:\n' f'```\n{self.browser_actions}\n```'
        )

    def __str__(self) -> str:
        ret = '**RelevantResearchRetrievalAction**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'RELEVANT RESEARCH RETRIEVAL ACTIONS: {self.browser_actions}'
        return ret
