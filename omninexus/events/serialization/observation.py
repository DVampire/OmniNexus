from omninexus.events.observation.agent import AgentStateChangedObservation
from omninexus.events.observation.browse import BrowserOutputObservation
from omninexus.events.observation.commands import (
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from omninexus.events.observation.delegate import AgentDelegateObservation
from omninexus.events.observation.empty import NullObservation
from omninexus.events.observation.error import ErrorObservation
from omninexus.events.observation.files import (
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
)
from omninexus.events.observation.observation import Observation
from omninexus.events.observation.reject import UserRejectObservation
from omninexus.events.observation.success import SuccessObservation

observations = (
    NullObservation,
    CmdOutputObservation,
    IPythonRunCellObservation,
    BrowserOutputObservation,
    FileReadObservation,
    FileWriteObservation,
    FileEditObservation,
    AgentDelegateObservation,
    SuccessObservation,
    ErrorObservation,
    AgentStateChangedObservation,
    UserRejectObservation,
)

OBSERVATION_TYPE_TO_CLASS = {
    observation_class.observation: observation_class  # type: ignore[attr-defined]
    for observation_class in observations
}


def observation_from_dict(observation: dict) -> Observation:
    observation = observation.copy()
    if 'observation' not in observation:
        raise KeyError(f"'observation' key is not found in {observation=}")
    observation_class = OBSERVATION_TYPE_TO_CLASS.get(observation['observation'])
    if observation_class is None:
        raise KeyError(
            f"'{observation['observation']=}' is not defined. Available observations: {OBSERVATION_TYPE_TO_CLASS.keys()}"
        )
    observation.pop('observation')
    observation.pop('message', None)
    content = observation.pop('content', '')
    extras = observation.pop('extras', {})
    return observation_class(content=content, **extras)
