"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json

from browsergym.core.action.highlevel import HighLevelActionSet
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)

from omninexus.agenthub.research_agent.modules.idea import (
    IdeaGenerationTool,
    RelevantResearchRetrievalTool,
)
from omninexus.agenthub.research_agent.modules.latex import (
    LatexAbstractTool,
    LatexConclusionTool,
    LatexDesignTool,
    LatexEmpiricalResultsTool,
    LatexIntroductionTool,
    LatexLimitationsAndFutureWorkTool,
    LatexMainTool,
    LatexMethodTool,
    LatexScriptsTool,
)
from omninexus.agenthub.research_agent.modules.project import (
    ProjectConfigurationTool,
    ProjectCriterionTool,
    ProjectDatasetTool,
    ProjectDesignTool,
    ProjectExperimentTool,
    ProjectLoggerTool,
    ProjectMetricTool,
    ProjectModelTool,
    ProjectOptimizerTool,
    ProjectRegistryTool,
    ProjectRunTool,
    ProjectSchedulerTool,
    ProjectTrainerTool,
    ProjectTransformTool,
    ProjectUtilsTool,
)
from omninexus.agenthub.research_agent.tools import (
    BrowserTool,
    CmdRunTool,
    FinishTool,
    IPythonTool,
    LLMBasedFileEditTool,
    StrReplaceEditorTool,
)
from omninexus.core.logger import omninexus_logger as logger
from omninexus.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    IdeaGenerationAction,
    IPythonRunCellAction,
    LatexAction,
    MessageAction,
    ProjectAction,
    RelevantResearchRetrievalAction,
)
from omninexus.events.tool import ToolCallMetadata


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, 'thought'):
        return action
    if thought:
        action.thought = thought
    return action


def response_to_actions(response: ModelResponse) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    assistant_msg = response.choices[0].message
    if assistant_msg.tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        thought = ''
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg['type'] == 'text':
                    thought += msg['text']

        # Process each tool call to omninexus action
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(
                    f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                ) from e
            if tool_call.function.name == 'execute_bash':
                action = CmdRunAction(**arguments)
            elif tool_call.function.name == 'execute_ipython_cell':
                action = IPythonRunCellAction(**arguments)
            elif tool_call.function.name == 'delegate_to_browsing_agent':
                action = AgentDelegateAction(
                    agent='BrowsingAgent',
                    inputs=arguments,
                )
            elif tool_call.function.name == 'finish':
                action = AgentFinishAction()
            elif tool_call.function.name == 'edit_file':
                action = FileEditAction(**arguments)
            elif tool_call.function.name == 'str_replace_editor':
                # We implement this in agent_skills, which can be used via Jupyter
                # convert tool_call.function.arguments to kwargs that can be passed to file_editor
                code = f'print(file_editor(**{arguments}))'
                logger.debug(
                    f'TOOL CALL: str_replace_editor -> file_editor with code: {code}'
                )
                action = IPythonRunCellAction(code=code, include_extra=False)
            elif tool_call.function.name == 'browser':
                action = BrowseInteractiveAction(browser_actions=arguments['code'])
            elif tool_call.function.name == 'relevant_research_retrieval':
                action = RelevantResearchRetrievalAction(
                    browser_actions=arguments['code']
                )
            elif tool_call.function.name == 'idea_generation':
                action = IdeaGenerationAction(**arguments)
            elif tool_call.function.name.startswith('project_'):
                action = ProjectAction(**arguments)
            elif tool_call.function.name.startswith('latex_'):
                action = LatexAction(**arguments)
            else:
                raise RuntimeError(f'Unknown tool call: {tool_call.function.name}')

            # We only add thought to the first action
            if i == 0:
                action = combine_thought(action, thought)
            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                model_response=response,
                total_calls_in_response=len(assistant_msg.tool_calls),
            )
            actions.append(action)
    else:
        actions.append(
            MessageAction(content=assistant_msg.content, wait_for_response=True)
        )

    assert len(actions) >= 1
    return actions


def get_tools(
    codeact_enable_browsing: bool = True,
    codeact_enable_llm_editor: bool = True,
    codeact_enable_jupyter: bool = True,
) -> list[ChatCompletionToolParam]:
    tools = [FinishTool, CmdRunTool, StrReplaceEditorTool]

    modules_idea = [
        RelevantResearchRetrievalTool,
        IdeaGenerationTool,
    ]

    modules_project = [
        ProjectConfigurationTool,
        ProjectCriterionTool,
        ProjectDatasetTool,
        ProjectDesignTool,
        ProjectExperimentTool,
        ProjectLoggerTool,
        ProjectMetricTool,
        ProjectModelTool,
        ProjectOptimizerTool,
        ProjectRegistryTool,
        ProjectRunTool,
        ProjectSchedulerTool,
        ProjectTrainerTool,
        ProjectTransformTool,
        ProjectUtilsTool,
    ]

    modules_latex = [
        LatexDesignTool,
        LatexAbstractTool,
        LatexIntroductionTool,
        LatexMethodTool,
        LatexEmpiricalResultsTool,
        LatexLimitationsAndFutureWorkTool,
        LatexConclusionTool,
        LatexScriptsTool,
        LatexMainTool,
    ]

    tools = modules_idea + modules_project + modules_latex + tools

    tools.append(BrowserTool)
    tools.append(IPythonTool)
    tools.append(LLMBasedFileEditTool)

    return tools
