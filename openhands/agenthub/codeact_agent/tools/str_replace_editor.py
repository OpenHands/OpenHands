# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon.
# Tag: Legacy-V0
import os
from pathlib import Path

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.agenthub.codeact_agent.tools.security_utils import (
    RISK_LEVELS,
    SECURITY_RISK_DESC,
)
from openhands.core.config.config_utils import DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX
from openhands.llm.tool_names import STR_REPLACE_EDITOR_TOOL_NAME

# Импорт хендлера для ноутбуков
try:
    from .notebook_handler import NotebookHandler
    _notebook_handler = NotebookHandler()
except ImportError:
    _notebook_handler = None

_DETAILED_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a text file, `view` displays the result of applying `cat -n`.
* NEW: Supports Jupyter Notebooks (.ipynb). It automatically strips heavy base64 images and truncates large outputs to prevent context overflow.
* The following binary file extensions can be viewed in Markdown format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"].
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
"""

_SHORT_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* Supports viewing Jupyter Notebooks (.ipynb) with automatic truncation of large outputs.
* The `undo_edit` command will revert the last edit made to the file at `path`
"""


def _get_workspace_mount_path_from_env(runtime_type: str | None = None) -> str:
    if runtime_type in ('local', 'cli'):
        sandbox_volumes = os.environ.get('SANDBOX_VOLUMES')
        if sandbox_volumes:
            mounts = sandbox_volumes.split(',')
            for mount in mounts:
                parts = mount.split(':')
                if len(parts) >= 2 and parts[1] == '/workspace':
                    host_path = os.path.abspath(parts[0])
                    return host_path
        return os.getcwd()
    return DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX


def create_str_replace_editor_tool(
    use_short_description: bool = False,
    workspace_mount_path_in_sandbox: str | None = None,
    runtime_type: str | None = None,
) -> ChatCompletionToolParam:
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
                        'enum': ['view', 'create', 'str_replace', 'insert', 'undo_edit'],
                        'type': 'string',
                    },
                    'path': {
                        'description': f'Absolute path to file, e.g. `{workspace_mount_path_in_sandbox}/file.py`.',
                        'type': 'string',
                    },
                    'file_text': {
                        'description': 'Content of the file to be created.',
                        'type': 'string',
                    },
                    'old_str': {
                        'description': 'The string in `path` to replace.',
                        'type': 'string',
                    },
                    'new_str': {
                        'description': 'The new string to replace or insert.',
                        'type': 'string',
                    },
                    'insert_line': {
                        'description': 'Line number AFTER which to insert `new_str`.',
                        'type': 'integer',
                    },
                    'view_range': {
                        'description': 'Optional line range, e.g. [1, 10].',
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
