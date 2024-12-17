"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_MODEL_DESCRIPTION = '''Generate a model module for the project.
* The assistant MUST first create a model.py file in src/model directory. The model.py file should contain the model class.
* The assistant MUST create a __init__.py file in the src/model directory to make it a package and import the model class in the __init__.py file.

Take the resnet34 for image classification as an example, the model.py and __init__.py files should look like this:

** model.py **
```python
import torch.nn as nn
from torchvision import models

from src.registry import MODEL


@MODEL.register_module(force=True)
class ResNet34(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """A ResNet-based image classification model.

        :param num_classes: Number of classes for classification.
        :param pretrained: If True, use a ResNet model pre-trained on ImageNet.
        """
        super(ResNet34, self).__init__()

        # Load a pre-trained ResNet model
        self.model = models.resnet34(
            pretrained=pretrained
        )  # Use resnet34 as an example

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Forward pass for the model.

        :param x: Input tensor of shape (batch_size, 3, H, W).
        :return: Output tensor of shape (batch_size, num_classes).
        """
        return self.model(x)

    def freeze_backbone(self):
        """Freeze the backbone (pre-trained ResNet layers) for feature extraction."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Keep the fully connected layer trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze the backbone (pre-trained ResNet layers) for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

```

** __init__.py **
```python
from .model import ResNet34

__all__ = ['ResNet34']
```

Note: Execute a bash command in the terminal to generate the model module. The command should be run in the root directory where the model module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
'''

ProjectModelTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_model',
        description=_MODEL_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the model module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
