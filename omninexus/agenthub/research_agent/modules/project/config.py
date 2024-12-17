"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_CONFIG_DESCRIPTION = '''Generate a configuration module for the project.
* The assistant MUST first create a config.py file in src/config directory. The cofig.py file should contain the config class.
* The assistant MUST create a __init__.py file in the src/config directory to make it a package and import the config class in the __init__.py file.

For example, the config.py and __init__.py files should look like this:

** config.py **
```python
import os
import shutil
from argparse import Namespace
from mmengine import Config

from src.logger import logger
from src.utils import assemble_project_path, init_before_training
from src.registry import CONFIG

@CONFIG.register_module(force=True)
class ConfigBuilder:
    def __init__(self, config_path: str, args: Namespace):
        self.config_path = config_path
        self.args = args
        self.config = Config.fromfile(filename=config_path)
        self._cfg_options = {}
        self._parse_args()

    def _parse_args(self):
        """Parse arguments and merge them into the config."""
        for item, value in self.args.__dict__.items():
            if item != 'config' and value is not None:
                self._cfg_options[item] = value
        self.config.merge_from_dict(self._cfg_options)

    def _set_exp_path(self):
        """Set up experiment directory path and handle removal if required."""
        self.config.exp_path = assemble_project_path(
            os.path.join(self.config.workdir, self.config.tag)
        )
        if self.config.if_remove is None:
            self.config.if_remove = input(f"| Arguments PRESS 'y' to REMOVE: {self.config.exp_path}? ") == 'y'

        if self.config.if_remove:
            shutil.rmtree(self.config.exp_path, ignore_errors=True)
            logger.info(f'| Arguments Remove work_dir: {self.config.exp_path}')
        else:
            logger.info(f'| Arguments Keep work_dir: {self.config.exp_path}')

        os.makedirs(self.config.exp_path, exist_ok=True)

    def _set_directories(self):
        """Set checkpoint and wandb paths."""
        self.config.checkpoint_path = os.path.join(self.config.exp_path, self.config.checkpoint_path)
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        logger.info(f'| Arguments Checkpoint path: {self.config.checkpoint_path}')

        self.config.wandb_path = os.path.join(self.config.exp_path, self.config.wandb_path)
        os.makedirs(self.config.wandb_path, exist_ok=True)
        logger.info(f'| Arguments Wandb path: {self.config.wandb_path}')

    def _init_seed(self):
        """Initialize the random seed for training."""
        init_before_training(self.config.seed)
        logger.info(f'| Arguments Seed: {self.config.seed}')

    def build(self) -> Config:
        """Build and return the final configuration."""
        self._set_exp_path()
        self._set_directories()
        self._init_seed()
        return self.config
```

** __init__.py **
```python
from .config import ConfigBuilder

__all__ = ['ConfigBuilder']
```

Note: Execute a bash command in the terminal to generate the configuration module. The command should be run in the root directory where the configuration module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
'''

ProjectConfigurationTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_configuration',
        description=_CONFIG_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the configuration module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
