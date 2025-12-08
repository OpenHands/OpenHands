# CodeAct Agent Framework

This folder is an implementation of OpenHands's main agent, the CodeAct Agent. It is based on ([CodeAct](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)), an idea of consolidating LLM agents' **act**ions into a unified **code** action space for both *simplicity* and *performance*.

## Overview

The CodeAct agent operates through a function calling interface. At each turn, the agent can:

1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
2. **CodeAct**: Execute actions through a set of well-defined tools:
   - Execute Linux `bash` commands with `execute_bash`
   - Run Python code in an [IPython](https://ipython.org/) environment with `execute_ipython_cell`
   - Interact with web browsers using `browser` and `fetch`
   - Edit files using `str_replace_editor` or `edit_file`

![image](https://github.com/OpenHands/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

## Built-in Tools

The agent provides several built-in tools:

### 1. `execute_bash`
- Execute any valid Linux bash command
- Handles long-running commands by running them in background with output redirection
- Supports interactive processes with STDIN input and process interruption
- Handles command timeouts with automatic retry in background mode

### 2. `execute_ipython_cell`
- Run Python code in an IPython environment
- Supports magic commands like `%pip`
- Variables are scoped to the IPython environment
- Requires defining variables and importing packages before use

### 3. `web_read` and `browser`
- `web_read`: Read and convert webpage content to markdown
- `browser`: Interact with webpages through Python code
- Supports common browser actions like navigation, clicking, form filling, scrolling
- Handles file uploads and drag-and-drop operations

### 4. `str_replace_editor`
- View, create and edit files through string replacement
- Persistent state across command calls
- File viewing with line numbers
- String replacement with exact matching
- Undo functionality for edits

### 5. `edit_file` (LLM-based)
- Edit files using LLM-based content generation
- Support for partial file edits with line ranges
- Handles large files by editing specific sections
- Append mode for adding content to files

## Configuration

Tools can be enabled/disabled through configuration parameters:
- `enable_browsing`: Enable browser interaction tools
- `enable_jupyter`: Enable IPython code execution
- `enable_llm_editor`: Enable LLM-based file editing (falls back to string replacement editor if disabled)

## Micro-agents

The agent includes specialized micro-agents for specific tasks:

1. **npm**: Handles npm package installation with non-interactive shell workarounds
2. **github**: Manages GitHub operations with API token support and PR creation guidelines
3. **flarglebargle**: Easter egg response handler

## Adding New Tools

The CodeAct agent uses a function calling interface based on `litellm`'s `ChatCompletionToolParam`. To add a new tool:

1. Define the tool in `function_calling.py`:
```python
MyTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='my_tool',
        description='Description of what the tool does and how to use it',
        parameters={
            'type': 'object',
            'properties': {
                'param1': {
                    'type': 'string',
                    'description': 'Description of parameter 1',
                },
                'param2': {
                    'type': 'integer',
                    'description': 'Description of parameter 2',
                },
            },
            'required': ['param1'],  # List required parameters here
        },
    ),
)
```

2. Add the tool to `get_tools()` in `function_calling.py`
3. Implement the corresponding action handler in the agent class

## Implementation Details

The agent is implemented in two main files:

1. `codeact_agent.py`: Core agent implementation with:
   - Message history management
   - Tool execution handling
   - State management
   - Action/observation processing

2. `function_calling.py`: Tool definitions and function calling interface with:
   - Tool parameter specifications
   - Tool descriptions and examples
   - Function calling response parsing
