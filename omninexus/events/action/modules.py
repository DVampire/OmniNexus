from dataclasses import dataclass
from typing import ClassVar

from omninexus.core.schema import ActionType
from omninexus.events.action.action import (
    Action,
    ActionConfirmationStatus,
    ActionSecurityRisk,
)


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


@dataclass
class IdeaGenerationAction(Action):
    command: str
    thought: str = ''
    blocking: bool = False
    # If False, the command will be run in a non-blocking / interactive way
    # The partial command outputs will be returned as output observation.
    # If True, the command will be run for max .timeout seconds.
    keep_prompt: bool = True
    # if True, the command prompt will be kept in the command output observation
    # Example of command output:
    # root@sandbox:~# ls
    # file1.txt
    # file2.txt
    # root@sandbox:~# <-- this is the command prompt

    hidden: bool = False
    action: str = ActionType.RUN
    runnable: ClassVar[bool] = True
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'Running command: {self.command}'

    def __str__(self) -> str:
        ret = f'**IdeaGenerationAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'COMMAND:\n{self.command}'
        return ret


@dataclass
class ProjectAction(Action):
    command: str
    thought: str = ''
    blocking: bool = False
    # If False, the command will be run in a non-blocking / interactive way
    # The partial command outputs will be returned as output observation.
    # If True, the command will be run for max .timeout seconds.
    keep_prompt: bool = True
    # if True, the command prompt will be kept in the command output observation
    # Example of command output:
    # root@sandbox:~# ls
    # file1.txt
    # file2.txt
    # root@sandbox:~# <-- this is the command prompt

    hidden: bool = False
    action: str = ActionType.RUN
    runnable: ClassVar[bool] = True
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'Generate project command: {self.command}'

    def __str__(self) -> str:
        ret = f'**ProjectAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'COMMAND:\n{self.command}'
        return ret


@dataclass
class LatexAction(Action):
    command: str
    thought: str = ''
    blocking: bool = False
    # If False, the command will be run in a non-blocking / interactive way
    # The partial command outputs will be returned as output observation.
    # If True, the command will be run for max .timeout seconds.
    keep_prompt: bool = True
    # if True, the command prompt will be kept in the command output observation
    # Example of command output:
    # root@sandbox:~# ls
    # file1.txt
    # file2.txt
    # root@sandbox:~# <-- this is the command prompt

    hidden: bool = False
    action: str = ActionType.RUN
    runnable: ClassVar[bool] = True
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'Generate latex command: {self.command}'

    def __str__(self) -> str:
        ret = f'**LatexAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'COMMAND:\n{self.command}'
        return ret
