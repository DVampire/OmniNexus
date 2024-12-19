from dataclasses import dataclass

from omninexus.core.schema import ObservationType
from omninexus.events.observation.observation import Observation


@dataclass
class IdeaGenerationObservation(Observation):
    """This data class represents the output of a command."""

    command_id: int
    command: str
    exit_code: int = 0
    hidden: bool = False
    observation: str = ObservationType.RUN
    interpreter_details: str = ''

    @property
    def error(self) -> bool:
        return self.exit_code != 0

    @property
    def message(self) -> str:
        return f'Idea Generation command `{self.command}` executed with exit code {self.exit_code}.'

    def __str__(self) -> str:
        return f'**IdeaGenerationObservation (source={self.source}, exit code={self.exit_code})**\n{self.content}'
