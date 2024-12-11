"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
import os.path

from browsergym.core.action.highlevel import HighLevelActionSet
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)

from omninexus.core.logger import omninexus_logger as logger
from omninexus.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    IPythonRunCellAction,
    MessageAction,
)
from omninexus.events.tool import ToolCallMetadata
from omninexus.utils.j2_utils import j2_to_dict

# Load the configuration for the CmdRun tool
cmd_run_config = j2_to_dict(
    os.path.join(os.path.dirname(__file__), 'descriptions', 'cmd_run.j2')
)
CmdRunTool = ChatCompletionToolParam(
    type=cmd_run_config['type'],
    function=ChatCompletionToolParamFunctionChunk(
        name=cmd_run_config['function']['name'],
        description=cmd_run_config['function']['description'],
        params=cmd_run_config['function']['parameters'],
    ),
)

# Load the configuration for the IPython tool
ipython_config = j2_to_dict(
    os.path.join(os.path.dirname(__file__), 'descriptions', 'ipython.j2')
)
IPythonTool = ChatCompletionToolParam(
    type=ipython_config['type'],
    function=ChatCompletionToolParamFunctionChunk(
        name=ipython_config['function']['name'],
        description=ipython_config['function']['description'],
        params=ipython_config['function']['parameters'],
    ),
)
