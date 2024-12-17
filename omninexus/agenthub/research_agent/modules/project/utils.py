"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_UTILS_DESCRIPTION = '''Generate an utils module for the project.
* The assistant MUST first create a utils.py file in src/utils directory. The utils.py file should contain the utility functions.
* The assistant MUST create a __init__.py file in the src/utils directory to make it a package and import the utils functions in the __init__.py file.

For example, the utils.py and __init__.py files should look like this:

** utils.py **
```python
import os
import random
import numpy as np
import torch


def get_project_root():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.dirname(path)  # get to parent, outside of project code path"
    return path


def assemble_project_path(path):
    """Assemble a path relative to the project root directory"""
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path


def init_before_training(seed=3407):
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)


def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            'float64': torch.float64,
            'float32': torch.float32,
            'float16': torch.float16,
            'fp32': torch.float32,
            'fp16': torch.float16,
            'half': torch.float16,
            'bf16': torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError
```

** __init__.py **
```python
from .utils import get_project_root, assemble_project_path, init_before_training, to_torch_dtype

__all__ = ['get_project_root', 'assemble_project_path', 'init_before_training', 'to_torch_dtype']
```

Note: Execute a bash command in the terminal to generate the utils module. The command should be run in the root directory where the utils module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
'''

ProjectUtilsTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_utils',
        description=_UTILS_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the utils module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
