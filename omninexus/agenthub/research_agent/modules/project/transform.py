"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_TRANSFORM_DESCRIPTION = '''Generate a transform module for the project.
* The assistant MUST first create a transform.py file in src/transform directory. The transform.py file should contain the transform class.
* The assistant MUST create a __init__.py file in the src/transform directory to make it a package and import the transform class in the __init__.py file.

Take image classification as an example, the transform.py and __init__.py files should look like this:

** transform.py **
```python
from torchvision import transforms

from src.logger import logger
from src.registry import TRANSFORM

@TRANSFORM.register_module(force=True)
class ImageTransform(transforms.Compose):
    def __init__(self, mode='train'):
        """A transform selector that dynamically constructs a transform pipeline based on the mode.
        :param mode: 'train', 'valid', or 'test'
        """
        if mode == 'train':
            logger.info('Using training transforms.')
            super().__init__(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                    ),  # Normalize to [-1, 1]
                ]
            )
        elif mode == 'valid':
            logger.info('Using validation transforms.')
            super().__init__(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                    ),  # Normalize to [-1, 1]
                ]
            )
        elif mode == 'test':
            logger.info('Using test transforms.')
            super().__init__(
                [
                    transforms.Resize((224, 224)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                    ),  # Normalize to [-1, 1]
                ]
            )
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. Use 'train', 'valid', or 'test'."
            )
```

** __init__.py **
```python
from .transform import ImageTransform

__all__ = ['ImageTransform']
```

Note: Execute a bash command in the terminal to generate the transform module. The command should be run in the root directory where the transform module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
'''

ProjectTransformTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_transform',
        description=_TRANSFORM_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the transform module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
