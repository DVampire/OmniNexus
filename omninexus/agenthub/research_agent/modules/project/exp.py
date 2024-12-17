"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_EXPERIMENT_DESCRIPTION = """Generate the experiment configuration file for the project.
* The assistant MUST first create an exp.py file in src/configs directory. The exp.py file should contain the experiment configuration parameters.

Take the resnet34 as an example, the exp.py file should look like this:

** exp.py **
```python
from copy import deepcopy

# fixed parameters
workdir = 'workdir'
tag = 'exp'
exp_path = f'{workdir}/{tag}'
project_name = 'research_project'
wandb_path = 'wandb'
checkpoint_path = 'checkpoints'
num_classes = 2
dtype = 'float32'
num_workers = 4

# configurable parameters
seed = 42
lr = 1e-3
epochs = int(1e3)
batch_size = 32
num_warmup_epochs = 10


transform = dict(type='ImageTransform', mode=None)

train_transform = deepcopy(transform)
train_transform['mode'] = 'train'

val_transform = deepcopy(transform)
val_transform['mode'] = 'valid'

test_transform = deepcopy(transform)
test_transform['mode'] = 'test'

dataset = dict(type='ImageDataset', image_dir=None, transform=None)

train_dataset = deepcopy(dataset)
train_dataset.update(
    {'image_dir': 'datasets/cat_and_dog/train', 'transform': train_transform}
)

val_dataset = deepcopy(dataset)
val_dataset.update(
    {'image_dir': 'datasets/cat_and_dog/test', 'transform': val_transform}
)

test_dataset = deepcopy(dataset)
test_dataset.update(
    {'image_dir': 'datasets/cat_and_dog/test', 'transform': test_transform}
)

model = dict(type='ResNet34', num_classes=num_classes)

criterion = dict(type='CrossEntropyLoss')

optimizer = dict(type='Adam', lr=lr, params=None)

scheduler = dict(
    type='CosineWithWarmupScheduler',
    optimizer=None,
    num_warmup_steps=None,
    num_training_steps=None,
)

metrics = [
    dict(type='Accuracy', topk=(1,)),
    dict(
        type='F1Score',
    ),
]

trainer = dict(
    type='Trainer',
    model=None,
    train_loader=None,
    valid_loader=None,
    test_loader=None,
    wandb_logger=None,
    accelerator=None,
    optimizer=None,
    scheduler=None,
    criterion=None,
    metrics=None,
    device=None,
    dtype=None,
    exp_path=None,
)

```

Note: Execute a bash command in the terminal to generate the exp.py. The command should be run in the root directory where the exp.py should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
"""

ProjectExperimentTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_experiment',
        description=_EXPERIMENT_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the experiment configuration file in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
