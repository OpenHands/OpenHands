from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_THINK_DESCRIPTION = """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.

Common use cases:
1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.
2. After receiving test results, use this tool to brainstorm ways to fix failing tests.
3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.
4. When designing a new feature, use this tool to think through architecture decisions and implementation details.
5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.

The tool simply logs your thought process for better transparency and does not execute any code or make changes."""

ThinkTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='think',
        description=_THINK_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'thought': {'type': 'string', 'description': 'The thought to log.'},
            },
            'required': ['thought'],
        },
    ),
)


_REFLECTION_DESCRIPTION = """Use the tool to reflect about one or a few previous steps. It will not obtain new information or make any changes to the repository, but just log the reflection.

Common use cases:
1. After a few rounds of executing commands or code, call this tool to reflect the progress, and assess which change(s) are needed.
2. If you see any execution errors or tool parse issues, call this tool immediately to come up with a plan to fix it
3. After receiving test results, use this tool to brainstorm ways to fix failing tests.

The tool simply logs your reflection process for better transparency and does not execute any code or make changes."""

ReflectionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='reflection',
        description=_REFLECTION_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'reflection': {'type': 'string', 'description': 'The reflection to log.'},
            },
            'required': ['reflection'],
        },
    ),
)
