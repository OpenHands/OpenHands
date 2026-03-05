from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

WARPGREP_TOOL_NAME = 'warpgrep_codebase_search'

_WARPGREP_DESCRIPTION = """Search the codebase using WarpGrep, an AI-powered code search subagent.

WarpGrep runs in an isolated context window. Instead of doing sequential grep/read calls (which pollutes your context with dead ends), WarpGrep does 8 parallel tool calls per turn across 4 turns, returning only the relevant code sections.

Use this for semantic searches like 'find the authentication middleware', 'where is rate limiting configured', or 'how are database connections managed'. For exact pattern matching, use bash with grep/rg instead."""

WarpGrepTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=WARPGREP_TOOL_NAME,
        description=_WARPGREP_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'A natural language description of what code to find.',
                },
            },
            'required': ['query'],
        },
    ),
)
