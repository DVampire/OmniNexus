from dotenv import load_dotenv

from omninexus.agenthub.micro.agent import MicroAgent
from omninexus.agenthub.micro.registry import all_microagents
from omninexus.controller.agent import Agent

load_dotenv()


from omninexus.agenthub import (  # noqa: E402
    browsing_agent,
    codeact_agent,
    delegator_agent,
    dummy_agent,
    research_agent
)

__all__ = [
    'codeact_agent',
    'delegator_agent',
    'dummy_agent',
    'browsing_agent',
    'research_agent',
]

for agent in all_microagents.values():
    name = agent['name']
    prompt = agent['prompt']

    anon_class = type(
        name,
        (MicroAgent,),
        {
            'prompt': prompt,
            'agent_definition': agent,
        },
    )

    Agent.register(name, anon_class)
