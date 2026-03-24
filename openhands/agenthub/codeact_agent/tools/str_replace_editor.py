# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# Tag: Legacy-V0
import os
from pathlib import Path
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk
from openhands.agenthub.codeact_agent.tools.security_utils import RISK_LEVELS, SECURITY_RISK_DESC
from openhands.core.config.config_utils import DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX
from openhands.llm.tool_names import STR_REPLACE_EDITOR_TOOL_NAME

try:
    from .notebook_handler import NotebookHandler
    _notebook_handler = NotebookHandler()
except ImportError:
    _notebook_handler = None

_DETAILED_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a text file, `view` displays the result of applying `cat -n`.
* NEW: Supports Jupyter Notebooks (.ipynb). It automatically strips heavy base64 images and truncates large outputs.
* The `undo_edit` command will revert the last edit made to the file at `path`
"""

_SHORT_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing and editing files. Supports .ipynb truncation."""

def _get_workspace_mount_path_from_env(runtime_type: str | None = None) -> str:
    if runtime_type in ('local', 'cli'):
        sandbox_volumes = os.environ.get('SANDBOX_VOLUMES')
        if sandbox_volumes:
            mounts = sandbox_volumes.split(',')
            for mount in mounts:
                parts = mount.split(':')
                if len(parts) >= 2 and parts[1] == '/workspace':
                    return os.path.abspath(parts[0])
        return os.getcwd()
    return DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX

def create_str_replace_editor_tool(
    use_short_description: bool = False,
    workspace_mount_path_in_sandbox: str | None = None,
    runtime_type: str | None = None,
) -> ChatCompletionToolParam:
    if workspace_mount_path_in_sandbox is None:
        workspace_mount_path_in_sandbox = _get_workspace_mount_path_from_env(runtime_type)
    
    description = _SHORT_STR_REPLACE_EDITOR_DESCRIPTION if use_short_description else _DETAILED_STR_REPLACE_EDITOR_DESCRIPTION
    
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name=STR_REPLACE_EDITOR_TOOL_NAME,
            description=description,
            parameters={
                'type': 'object',
                'properties': {
                    'command': {
                        'description': 'Allowed options: `view`, `create`, `str_replace`, `insert`, `undo_edit`.',
                        'enum': ['view', 'create', 'str_replace', 'insert', 'undo_edit'],
                        'type': 'string',
                    },
                    'path': {'description': 'Absolute path to file.', 'type': 'string'},
                    'file_text': {'description': 'Content for create.', 'type': 'string'},
                    'old_str': {'description': 'String to replace.', 'type': 'string'},
                    'new_str': {'description': 'New string.', 'type': 'string'},
                    'insert_line': {'description': 'Line number to insert after.', 'type': 'integer'},
                    'view_range': {'items': {'type': 'integer'}, 'type': 'array'},
                    'security_risk': {'type': 'string', 'description': SECURITY_RISK_DESC, 'enum': RISK_LEVELS},
                },
                'required': ['command', 'path', 'security_risk'],
            },
        ),
    )
