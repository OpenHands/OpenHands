# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_SEARCH_ENTITY_DESCRIPTION = """
Searches the codebase to retrieve the complete implementations of specified entities based on the provided entity names.
The tool can handle specific entity queries such as function names, class names, or file paths.

**Usage Example:**
# Search for a specific function implementation
get_entity_contents(['src/my_file.py:MyClass.func_name'])

# Search for a file's complete content
get_entity_contents(['src/my_file.py'])

**Entity Name Format:**
- To specify a function or class, use the format: `file_path:QualifiedName`
  (e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum').
- To search for a file's content, use only the file path (e.g., 'src/my_file.py').
"""

SearchEntityTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='get_entity_contents',
        description=_SEARCH_ENTITY_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'entity_names': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'A list of entity names to query. Each entity name can represent a function, class, or file. '
                        "For functions or classes, the format should be 'file_path:QualifiedName' "
                        "(e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum'). "
                        "For files, use just the file path (e.g., 'src/my_file.py')."
                    ),
                }
            },
            'required': ['entity_names'],
        },
    ),
)


_SEARCH_REPO_DESCRIPTION = """Searches the codebase to retrieve relevant code snippets based on given queries(terms or line numbers).
** Note:
- Either `search_terms` or `line_nums` must be provided to perform a search.
- If `search_terms` are provided, it searches for code snippets based on each term:
- If `line_nums` is provided, it searches for code snippets around the specified lines within the file defined by `file_path_or_pattern`.

** Example Usage:
# Search for code content contain keyword `order`, `bill`
search_code_snippets(search_terms=["order", "bill"])

# Search for a class
search_code_snippets(search_terms=["MyClass"])

# Search for context around specific lines (10 and 15) within a file
search_code_snippets(line_nums=[10, 15], file_path_or_pattern='src/example.py')
"""

SearchRepoTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_code_snippets',
        description=_SEARCH_REPO_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'search_terms': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'A list of names, keywords, or code snippets to search for within the codebase. '
                    'This can include potential function names, class names, or general code fragments. '
                    'Either `search_terms` or `line_nums` must be provided to perform a search.',
                },
                'line_nums': {
                    'type': 'array',
                    'items': {'type': 'integer'},
                    'description': 'Specific line numbers to locate code snippets within a specified file. '
                    'Must be used alongside a valid `file_path_or_pattern`. '
                    'Either `line_nums` or `search_terms` must be provided to perform a search.',
                },
                'file_path_or_pattern': {
                    'type': 'string',
                    'description': 'A glob pattern or specific file path used to filter search results '
                    'to particular files or directories. Defaults to "**/*.py", meaning all Python files are searched by default. '
                    'If `line_nums` are provided, this must specify a specific file path.',
                    'default': '**/*.py',
                },
            },
            'required': [],
        },
    ),
)
