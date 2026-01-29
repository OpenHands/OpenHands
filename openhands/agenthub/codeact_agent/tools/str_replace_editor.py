# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import os

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.agenthub.codeact_agent.tools.security_utils import (
    RISK_LEVELS,
    SECURITY_RISK_DESC,
)
from openhands.core.config.config_utils import DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX
from openhands.llm.tool_names import STR_REPLACE_EDITOR_TOOL_NAME

_DETAILED_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a text file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The following binary file extensions can be viewed in Markdown format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"]. IT DOES NOT HANDLE IMAGES.
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
* This tool can be used for creating and editing files in plain-text format.


Before using this tool:
1. Use the view tool to understand the file's contents and context
2. Verify the directory path is correct (only applicable when creating new files):
   - Use the view tool to verify the parent directory exists and is the correct location

When making edits:
   - Ensure the edit results in idiomatic, correct code
   - Do not leave the code in a broken state
   - Always use absolute file paths (starting with /)

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.

2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
   - Include sufficient context before and after the change point (3-5 lines recommended)
   - If not unique, the replacement will not be performed

3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.
"""

_SHORT_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""


def _get_workspace_mount_path_from_env(runtime_type: str | None = None) -> str:
    """Get the workspace mount path from SANDBOX_VOLUMES environment variable.

    For LocalRuntime and CLIRuntime, returns the host path from SANDBOX_VOLUMES.
    For other runtimes, returns the default container path (/workspace).

    Args:
        runtime_type: The runtime type ('local', 'cli', 'docker', etc.)

    Returns:
        The workspace mount path in sandbox, defaults to '/workspace' if not found.
    """
    # For LocalRuntime/CLIRuntime, try to get host path from SANDBOX_VOLUMES
    if runtime_type in ('local', 'cli'):
        sandbox_volumes = os.environ.get('SANDBOX_VOLUMES')
        if sandbox_volumes:
            # Split by commas to handle multiple mounts
            mounts = sandbox_volumes.split(',')

            # Check if any mount explicitly targets /workspace
            for mount in mounts:
                parts = mount.split(':')
                if len(parts) >= 2 and parts[1] == '/workspace':
                    host_path = os.path.abspath(parts[0])
                    return host_path

        # Fallback for local/CLI runtimes when SANDBOX_VOLUMES is not set:
        # Use current working directory as it's likely the workspace root
        return os.getcwd()

    # For all other runtimes (docker, remote, etc.), use default container path
    return DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX


def create_str_replace_editor_tool(
    use_short_description: bool = False,
    workspace_mount_path_in_sandbox: str | None = None,
    runtime_type: str | None = None,
) -> ChatCompletionToolParam:
    # If no workspace path is provided, try to get it from environment
    if workspace_mount_path_in_sandbox is None:
        workspace_mount_path_in_sandbox = _get_workspace_mount_path_from_env(
            runtime_type
        )

    description = (
        _SHORT_STR_REPLACE_EDITOR_DESCRIPTION
        if use_short_description
        else _DETAILED_STR_REPLACE_EDITOR_DESCRIPTION
    )
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name=STR_REPLACE_EDITOR_TOOL_NAME,
            description=description,
            parameters={
                'type': 'object',
                'properties': {
                    'command': {
                        'description': 'The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.',
                        'enum': [
                            'view',
                            'create',
                            'str_replace',
                            'insert',
                            'undo_edit',
                        ],
                        'type': 'string',
                    },
                    'path': {
                        'description': f'Absolute path to file or directory, e.g. `{workspace_mount_path_in_sandbox}/file.py` or `{workspace_mount_path_in_sandbox}`.',
                        'type': 'string',
                    },
                    'file_text': {
                        'description': 'Required parameter of `create` command, with the content of the file to be created.',
                        'type': 'string',
                    },
                    'old_str': {
                        'description': 'Required parameter of `str_replace` command containing the string in `path` to replace.',
                        'type': 'string',
                    },
                    'new_str': {
                        'description': 'Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.',
                        'type': 'string',
                    },
                    'insert_line': {
                        'description': 'Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.',
                        'type': 'integer',
                    },
                    'view_range': {
                        'description': 'Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.',
                        'items': {'type': 'integer'},
                        'type': 'array',
                    },
                    'security_risk': {
                        'type': 'string',
                        'description': SECURITY_RISK_DESC,
                        'enum': RISK_LEVELS,
                    },
                },
                'required': ['command', 'path', 'security_risk'],
            },
        ),
    )
