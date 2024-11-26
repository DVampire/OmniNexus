from dataclasses import dataclass

from omninexus.events.event import Event


@dataclass
class Observation(Event):
    content: str
