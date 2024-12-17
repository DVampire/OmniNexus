"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_OPTIMIZER_DESCRIPTION = """Generate an optimizer module for the project.
* The assistant MUST first create a optimizer.py file in src/optimizer directory. The optimizer.py file should contain the optimizer class.
* The assistant MUST create a __init__.py file in the src/optimizer directory to make it a package and import the optimizer class in the __init__.py file.

For example, the optimizer.py and __init__.py files should look like this:

** optimizer.py **
```python
from torch.optim import Adam
from transformers.optimization import AdamW

from rp.registry import OPTIMIZER

OPTIMIZER.register_module(name='AdamW', module=AdamW)
OPTIMIZER.register_module(name='Adam', module=Adam)
```

** __init__.py **
```python
from .optimizer import Adam, AdamW

__all__ = ['Adam', 'AdamW']
```

Note: Execute a bash command in the terminal to generate the optimizer module. The command should be run in the root directory where the optimizer module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
"""

ProjectOptimizerTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_optimizer',
        description=_OPTIMIZER_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the optimizer module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
