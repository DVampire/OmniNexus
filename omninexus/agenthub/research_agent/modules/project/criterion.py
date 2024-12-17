"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_CRITERION_DESCRIPTION = """Generate a criterion module for the project.
* The assistant MUST first create a criterion.py file in src/criterion directory. The criterion.py file should contain the criterion class.
* The assistant MUST create a __init__.py file in the src/criterion directory to make it a package and import the criterion class in the __init__.py file.

For example, the criterion.py and __init__.py files should look like this:

** criterion.py **
```python
from torch.nn import CrossEntropyLoss, MSELoss

from src.registry import CRITERION

CRITERION.register_module(name='MSELoss', module=MSELoss)
CRITERION.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
```

** __init__.py **
```python
from .criterion import MSELoss, CrossEntropyLoss

__all__ = ['MSELoss', 'CrossEntropyLoss']
```

Note: Execute a bash command in the terminal to generate the criterion module. The command should be run in the root directory where the criterion module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
"""

ProjectCriterionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_criterion',
        description=_CRITERION_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the criterion module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
