from dataclasses import dataclass

from omninexus.core.schema import ActionType
from omninexus.events.action.action import Action, ActionSecurityRisk


@dataclass
class IdeaGenerationAction(Action):
    retrieved_relevant_researches: str
    wait_for_response: bool = False
    action: str = ActionType.MESSAGE
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return self.retrieved_relevant_researches

    def __str__(self) -> str:
        ret = f'**IdeaGenerationAction** (source={self.source})\n'
        ret += f'CONTENT: {self.retrieved_relevant_researches}'
        return ret
