"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_REGISTRY_DESCRIPTION = """Generate a registry to register all modules.
* The assistant MUST refer to the usage of mmengine.registry.Registry to register each module of the project.

For example, the registry.py file should look like this:
```python
from mmengine.registry import Registry

DATASET = Registry('dataset', locations=['src.dataset']) # Register the dataset module
TRANSFORM = Registry('transform', locations=['src.transform']) # Register the transform module

MODEL = Registry('model', locations=['src.model']) # Register the model module
OPTIMIZER = Registry('optimizer', locations=['src.optimizer']) # Register the optimizer module
SCHEDULER = Registry('scheduler', locations=['src.scheduler']) # Register the scheduler module
CRITERION = Registry('criterion', locations=['src.criterion']) # Register the criterion module

LOGGER = Registry('logger', locations=['src.logger']) # Register the logger module

METRIC = Registry('metric', locations=['src.metric']) # Register the metric module
TRAINER = Registry('trainer', locations=['src.trainer']) # Register the trainer module

CONFIG = Registry('config', locations=['src.config']) # Register the config module
```

Note: Execute a bash command in the terminal to generate the registry.py file. The command should be run in the root directory where the registry.py file should be created.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
"""

ProjectRegistryTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_registry',
        description=_REGISTRY_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the registry.py in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
