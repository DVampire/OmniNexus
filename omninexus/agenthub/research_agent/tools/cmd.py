from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_BASH_DESCRIPTION = """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command like `C-c` (Ctrl+C) to interrupt the process.
"""

CmdRunTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='execute_bash',
        description=_BASH_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
