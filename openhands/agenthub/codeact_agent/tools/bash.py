# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.agenthub.codeact_agent.tools.prompt import refine_prompt
from openhands.agenthub.codeact_agent.tools.security_utils import (
    RISK_LEVELS,
    SECURITY_RISK_DESC,
)
from openhands.llm.tool_names import EXECUTE_BASH_TOOL_NAME

_DETAILED_BASH_DESCRIPTION = """Execute a bash command in the terminal within a persistent shell session.


### Command Execution
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
* Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command (see section below for details)
* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts or commands in this environment. The runtime may not support them and can cause unusable shell sessions. If you want to run multi-line bash commands, write the commands to a file and then run it, instead.

### Long-running Commands
* For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
* For commands that may run for a long time (e.g. installation or testing commands), or commands that run for a fixed amount of time (e.g. sleep), you should set the "timeout" parameter of your function call to an appropriate value.
* If a bash command returns exit code `-1`, this means the process hit the soft timeout and is not yet finished. By setting `is_input` to `true`, you can:
  - Send empty `command` to retrieve additional logs
  - Send text (set `command` to the text) to STDIN of the running process
  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process
  - If you do C-c, you can re-start the process with a longer "timeout" parameter to let it run to completion

### Best Practices
* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

### Output Handling
* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.
"""

_SHORT_BASH_DESCRIPTION = """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`. For commands that need to run for a specific duration, you can set the "timeout" argument to specify a hard timeout in seconds.
* Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, the assistant can interact with the running process and send empty `command` to retrieve any additional logs, or send additional text (set `command` to the text) to STDIN of the running process, or send command like `C-c` (Ctrl+C), `C-d` (Ctrl+D), `C-z` (Ctrl+Z) to interrupt the process.
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."""


def create_cmd_run_tool(
    use_short_description: bool = False,
) -> ChatCompletionToolParam:
    description = (
        _SHORT_BASH_DESCRIPTION if use_short_description else _DETAILED_BASH_DESCRIPTION
    )
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name=EXECUTE_BASH_TOOL_NAME,
            description=refine_prompt(description),
            parameters={
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': refine_prompt(
                            'The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.'
                        ),
                    },
                    'is_input': {
                        'type': 'string',
                        'description': refine_prompt(
                            'If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.'
                        ),
                        'enum': ['true', 'false'],
                    },
                    'timeout': {
                        'type': 'number',
                        'description': 'Optional. Sets a hard timeout in seconds for the command execution. If not provided, the command will use the default soft timeout behavior.',
                    },
                    'security_risk': {
                        'type': 'string',
                        'description': SECURITY_RISK_DESC,
                        'enum': RISK_LEVELS,
                    },
                },
                'required': ['command', 'security_risk'],
            },
        ),
    )
