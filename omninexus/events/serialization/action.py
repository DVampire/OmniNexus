from omninexus.core.exceptions import LLMMalformedActionError
from omninexus.events.action.action import Action
from omninexus.events.action.agent import (
    AgentDelegateAction,
    AgentFinishAction,
    AgentRejectAction,
    ChangeAgentStateAction,
)
from omninexus.events.action.browse import BrowseInteractiveAction, BrowseURLAction
from omninexus.events.action.commands import (
    CmdRunAction,
    IPythonRunCellAction,
)
from omninexus.events.action.empty import NullAction
from omninexus.events.action.files import (
    FileEditAction,
    FileReadAction,
    FileWriteAction,
)
from omninexus.events.action.message import MessageAction

actions = (
    NullAction,
    CmdRunAction,
    IPythonRunCellAction,
    BrowseURLAction,
    BrowseInteractiveAction,
    FileReadAction,
    FileWriteAction,
    FileEditAction,
    AgentFinishAction,
    AgentRejectAction,
    AgentDelegateAction,
    ChangeAgentStateAction,
    MessageAction,
)

ACTION_TYPE_TO_CLASS = {action_class.action: action_class for action_class in actions}  # type: ignore[attr-defined]


def action_from_dict(action: dict) -> Action:
    if not isinstance(action, dict):
        raise LLMMalformedActionError('action must be a dictionary')
    action = action.copy()
    if 'action' not in action:
        raise LLMMalformedActionError(f"'action' key is not found in {action=}")
    if not isinstance(action['action'], str):
        raise LLMMalformedActionError(
            f"'{action['action']=}' is not defined. Available actions: {ACTION_TYPE_TO_CLASS.keys()}"
        )
    action_class = ACTION_TYPE_TO_CLASS.get(action['action'])
    if action_class is None:
        raise LLMMalformedActionError(
            f"'{action['action']=}' is not defined. Available actions: {ACTION_TYPE_TO_CLASS.keys()}"
        )
    args = action.get('args', {})
    # Remove timestamp from args if present
    timestamp = args.pop('timestamp', None)

    # compatibility for older event streams
    # is_confirmed has been renamed to confirmation_state
    is_confirmed = args.pop('is_confirmed', None)
    if is_confirmed is not None:
        args['confirmation_state'] = is_confirmed

    # images_urls has been renamed to image_urls
    if 'images_urls' in args:
        args['image_urls'] = args.pop('images_urls')

    # keep_prompt has been deprecated in https://github.com/All-Hands-AI/OpenHands/pull/4881
    if 'keep_prompt' in args:
        args.pop('keep_prompt')

    try:
        decoded_action = action_class(**args)
        if 'timeout' in action:
            decoded_action.timeout = action['timeout']

        # Set timestamp if it was provided
        if timestamp:
            decoded_action._timestamp = timestamp

    except TypeError as e:
        raise LLMMalformedActionError(
            f'action={action} has the wrong arguments: {str(e)}'
        )
    return decoded_action
