from dataclasses import dataclass

from omninexus.core.schema import ObservationType
from omninexus.events.observation.observation import Observation


@dataclass
class AgentStateChangedObservation(Observation):
    """This data class represents the result from delegating to another agent"""

    agent_state: str
    observation: str = ObservationType.AGENT_STATE_CHANGED

    @property
    def message(self) -> str:
        return ''


@dataclass
class AgentCondensationObservation(Observation):
    """The output of a condensation action."""

    observation: str = ObservationType.CONDENSE

    @property
    def message(self) -> str:
        return self.content
