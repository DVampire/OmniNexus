from dataclasses import dataclass

from omninexus.core.schema import ActionType
from omninexus.events.action.action import Action


@dataclass
class NullAction(Action):
    """An action that does nothing."""

    action: str = ActionType.NULL

    @property
    def message(self) -> str:
        return 'No action'
