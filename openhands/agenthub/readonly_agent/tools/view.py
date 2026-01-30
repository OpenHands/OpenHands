# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_VIEW_DESCRIPTION = """Reads a file or list directories from the local filesystem.
* The path parameter must be an absolute path, not a relative path.
* If `path` is a file, `view` displays the result of applying `cat -n`; if `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep.
* You can optionally specify a line range to view (especially handy for long files), but it's recommended to read the whole file by not providing this parameter.
* For image files, the tool will display the image for you.
* For large files that exceed the display limit:
  - The output will be truncated and marked with `<response clipped>`
  - Use the `view_range` parameter to view specific sections after the truncation point
"""

ViewTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='view',
        description=_VIEW_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The absolute path to the file to read or directory to list',
                },
                'view_range': {
                    'description': 'Optional parameter of `view` command when `path` points to a *file*. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.',
                    'items': {'type': 'integer'},
                    'type': 'array',
                },
            },
            'required': ['path'],
        },
    ),
)
