# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_GREP_DESCRIPTION = """Fast content search tool.
* Searches file contents using regular expressions
* Supports full regex syntax (eg. "log.*Error", "function\\s+\\w+", etc.)
* Filter files by pattern with the include parameter (eg. "*.js", "*.{ts,tsx}")
* Returns matching file paths sorted by modification time.
* Only the first 100 results are returned. Consider narrowing your search with stricter regex patterns or provide path parameter if you need more results.
* Use this tool when you need to find files containing specific patterns
* When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
"""

GrepTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='grep',
        description=_GREP_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'The regex pattern to search for in file contents',
                },
                'path': {
                    'type': 'string',
                    'description': 'The directory (absolute path) to search in. Defaults to the current working directory.',
                },
                'include': {
                    'type': 'string',
                    'description': 'Optional file pattern to filter which files to search (e.g., "*.js", "*.{ts,tsx}")',
                },
            },
            'required': ['pattern'],
        },
    ),
)
