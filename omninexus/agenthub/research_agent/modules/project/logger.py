"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LOGGER_DESCRIPTION = '''Generate a logger module for the project to handle logging functionality.
* The assistant MUST first create a logger.py file in src/logger directory. The logger module should contain a logger object that can be used to log messages to the console.
* The assistant MUST create a wandb_logger.py file in the src/logger directory. The wandb_logger module should contain a logger object that can be used to log messages to Weights & Biases.
* The assistant MUST create a __init__.py file in the src/logger directory to make the logger module importable.

For example, the logger.py, wandb_logger.py, and __init__.py files should look like this:

** logger.py **
```python
import logging

from src.registry import LOGGER

@LOGGER.register_module(force=True)
class Logger(logging.Logger):
    def __init__(self, name='logger', level=logging.INFO):
        """A custom logger class for enhanced console logging with colored output.

        :param name: Name of the logger.
        :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        super().__init__(name, level)

        # Formatter for log messages
        formatter = logging.Formatter(
            fmt='\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        # Console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        # Add the handler to the logger instance
        self.addHandler(console_handler)
        self.propagate = (
            False  # Prevent duplicate logs from propagating to the root logger
        )

logger = Logger(name='logger')
```

** wandb_logger.py **
```python
import torch
import wandb

from src.logger import research_logger as logger
from src.registry import LOGGER


@LOGGER.register_module(force=True)
class WandbLogger:
    def __init__(self, log_dir, project_name, run_name=None, config=None):
        """A class for managing wandb logging and experiment tracking.

        :param log_dir: The directory to save wandb logs.
        :param project_name: The name of the wandb project.
        :param run_name: Optional, the name of the specific run.
        :param config: Optional, a dictionary containing configuration parameters.
        """
        self.log_dir = log_dir
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.run = None

    def initialize(self):
        """Initialize a wandb run with the given parameters."""
        self.run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=self.config,
            dir=self.log_dir,
        )
        logger.info(f'Wandb run initialized: {self.run_name or self.run.name}')

    def log(self, metrics, step=None):
        """Log metrics to wandb.

        :param metrics: A dictionary of metrics to log (e.g., {'loss': 0.1, 'accuracy': 0.9}).
        :param step: Optional, the step at which the metrics are logged.
        """
        if self.run is None:
            raise RuntimeError(
                'Wandb run has not been initialized. Call `initialize()` first.'
            )
        wandb.log(metrics, step=step)

    def save_model(self, model, model_path):
        """Save a model file and upload it to wandb.

        :param model: The model to save.
        :param model_path: The local path to save the model.
        """
        if self.run is None:
            raise RuntimeError(
                'Wandb run has not been initialized. Call `initialize()` first.'
            )

        # Save the model locally
        torch.save(model.state_dict(), model_path)
        logger.info(f'Model saved locally at {model_path}')

        # Upload to wandb
        wandb.save(model_path)
        logger.info(f'Model uploaded to wandb: {model_path}')

    def finish(self):
        """Finish the current wandb run."""
        if self.run is not None:
            self.run.finish()
            logger.info('Wandb run finished.')
        else:
            logger.info('Wandb run was not initialized.')
```

** __init__.py **
```python
from .logger import logger, Logger
from .wandb_logger import WandbLogger

__all__ = ['logger', 'Logger', 'WandbLogger']
```

Note: Execute a bash command in the terminal to generate the logger module. The command should be run in the root directory where the logger module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
'''

ProjectLoggerTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_logger',
        description=_LOGGER_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the logger module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
