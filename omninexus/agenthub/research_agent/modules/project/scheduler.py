"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_SCHEDULER_DESCRIPTION = """Generate a scheduler module for the project.
* The assistant MUST first create a scheduler.py file in src/scheduler directory. The scheduler.py file should contain the scheduler class.
* The assistant MUST create a __init__.py file in the src/scheduler directory to make it a package and import the scheduler class in the __init__.py file.

For example, the scheduler.py and __init__.py files should look like this:

** scheduler.py **
```python

```

** __init__.py **
```python
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from src.registry import SCHEDULER


@SCHEDULER.register_module(force=True)
class CosineWithWarmupScheduler:
    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        num_cycles=0.5,
        last_epoch=-1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self._scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    def __str__(self):
        return f'CosineWithWarmupScheduler(num_warmup_steps={self.num_warmup_steps}, num_training_steps={self.num_training_steps})'

    def __getattr__(self, name):
        return getattr(self._scheduler, name)


@SCHEDULER.register_module(force=True)
class LinearWithWarmupScheduler:
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self._scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

    def __str__(self):
        return f'LinearWithWarmupScheduler(num_warmup_steps={self.num_warmup_steps}, num_training_steps={self.num_training_steps})'

    def __getattr__(self, name):
        return getattr(self._scheduler, name)


@SCHEDULER.register_module(force=True)
class ConstantWithWarmupScheduler:
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self._scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

    def __str__(self):
        return f'ConstantWithWarmupScheduler(num_warmup_steps={self.num_warmup_steps}, num_training_steps={self.num_training_steps})'

    def __getattr__(self, name):
        return getattr(self._scheduler, name)


@SCHEDULER.register_module(force=True)
class PolynomialDecayWithWarmupScheduler:
    def __init__(
        self, optimizer, num_warmup_steps, num_training_steps, power=1.0, last_epoch=-1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self._scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            last_epoch=last_epoch,
        )

    def __str__(self):
        return f'PolynomialDecayWithWarmupScheduler(num_warmup_steps={self.num_warmup_steps}, num_training_steps={self.num_training_steps})'

    def __getattr__(self, name):
        return getattr(self._scheduler, name)
```

** __init__.py **
```python
from .scheduler import CosineWithWarmupScheduler, LinearWithWarmupScheduler, ConstantWithWarmupScheduler, PolynomialDecayWithWarmupScheduler

__all__ = [
    'CosineWithWarmupScheduler',
    'LinearWithWarmupScheduler',
    'ConstantWithWarmupScheduler',
    'PolynomialDecayWithWarmupScheduler',
]
```

Note: Execute a bash command in the terminal to generate the scheduler module. The command should be run in the root directory where the scheduler module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
"""

ProjectSchedulerTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_scheduler',
        description=_SCHEDULER_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the scheduler module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
