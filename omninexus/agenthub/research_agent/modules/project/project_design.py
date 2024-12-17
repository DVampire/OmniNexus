"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_PROJECT_DESIGN_DESCRIPTION = """Design the code organization structure of the research project, along with a bash command to generate the it directly in the terminal.
* The assistant should organize the project structure by leveraging mmengine, ensuring the project is well-modularized.
* The assistant should name the project appropriately based on the task, and its directory structure is as follows:
[project_name]                   # Root directory of the project, [project_name] should be replaced with the actual project name.
├── src                          # Main package containing core functionality
│   ├── config                   # Module for managing configurations.
│   ├── criterion                # Module for loss functions.
│   ├── dataset                  # Module for dataset handling.
│   ├── logger                   # Module for logging.
│   ├── metric                   # Module for evaluation metrics.
│   ├── model                    # Module for defining neural network architectures.
│   ├── optimizer                # Module for optimizers.
│   ├── scheduler                # Module for learning rate schedulers.
│   ├── trainer                  # Module for training logic.
│   ├── transform                # Module for data transformation.
│   ├── utils                    # Subpackage for general-purpose utility functions.
│   ├── __init__.py              # Init file for the src directory
│   └── registry.py              # Module for registering components, e.g., models, optimizers, schedulers, etc.
├── configs                      # Directory for configuration files, e.g., for hyperparameters, experiment settings, etc.
├── datasets                     # Directory for datasets (ignored; provided by the user)
├── test                         # Directory for unit tests.
├── run.py                       # Main entry point for running the project.
├── README.md                    # Project documentation.
├── requirements.txt             # File containing the required Python packages.
└── .gitignore                   # File for ignoring files and directories in Git.
* The assistant should ensure that the project structure is well-documented and follows best practices for code organization.

Note: Execute a bash command in the terminal to generate the project structure. The command should be run in the root directory where the project structure should be created.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
"""

ProjectDesignTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_design',
        description=_PROJECT_DESIGN_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the project structure in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
