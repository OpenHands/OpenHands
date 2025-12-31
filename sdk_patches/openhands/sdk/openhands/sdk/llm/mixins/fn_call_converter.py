"""Convert function calling messages to non-function calling messages and vice versa.

This will inject prompts so that models that doesn't support function calling
can still be used with function calling agents.

We follow format from: https://docs.litellm.ai/docs/completion/function_call
"""  # noqa: E501

import copy
import json
import re
import sys
from collections.abc import Iterable
from typing import Any, Literal, NotRequired, TypedDict, cast

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.sdk.llm.exceptions import (
    FunctionCallConversionError,
    FunctionCallValidationError,
)


class CacheControl(TypedDict):
    type: Literal["ephemeral"]


class TextPart(TypedDict):
    type: Literal["text"]
    text: str
    cache_control: NotRequired[CacheControl]


Content = str | list[TextPart]

TERMINAL_TOOL_NAME = "terminal"
STR_REPLACE_EDITOR_TOOL_NAME = "file_editor"
BROWSER_TOOL_NAME = "browser"
FINISH_TOOL_NAME = "finish"
LLM_BASED_EDIT_TOOL_NAME = "edit_file"
TASK_TRACKER_TOOL_NAME = "task_tracker"

# Inspired by: https://docs.together.ai/docs/llama-3-function-calling#function-calling-w-llama-31-70b
MISSING_DESCRIPTION_PLACEHOLDER = "No description provided"
SCHEMA_INDENT_STEP = 2
SCHEMA_UNION_KEYS = ("anyOf", "oneOf", "allOf")


system_message_suffix_TEMPLATE = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""  # noqa: E501

STOP_WORDS = ["</function"]


def refine_prompt(prompt: str) -> str:
    if sys.platform == "win32":
        return prompt.replace("bash", "powershell")
    return prompt


# NOTE: we need to make sure these examples are always in-sync with the tool
# interface designed in openhands/agenthub/agent/function_calling.py

# Example snippets for each tool
TOOL_EXAMPLES = {
    "bash": {
        "check_dir": """
ASSISTANT: Sure! Let me first check the current directory:
<function=terminal>
<parameter=command>
pwd && ls
</parameter>
</function>

USER: EXECUTION RESULT of [terminal]:
/workspace
openhands@runtime:~/workspace$
""",  # noqa: E501
        "run_server": """
ASSISTANT:
Let me run the Python file for you:
<function=terminal>
<parameter=command>
python3 app.py > server.log 2>&1 &
</parameter>
</function>

USER: EXECUTION RESULT of [terminal]:
[1] 121
[1]+  Exit 1                  python3 app.py > server.log 2>&1

ASSISTANT:
Looks like the server was running with PID 121 then crashed. Let me check the server log:
<function=terminal>
<parameter=command>
cat server.log
</parameter>
</function>

USER: EXECUTION RESULT of [terminal]:
Traceback (most recent call last):
  File "/workspace/app.py", line 2, in <module>
    from flask import Flask
ModuleNotFoundError: No module named 'flask'

ASSISTANT:
Looks like the server crashed because the `flask` module is not installed. Let me install the `flask` module for you:
<function=terminal>
<parameter=command>
pip3 install flask
</parameter>
</function>

USER: EXECUTION RESULT of [terminal]:
Defaulting to user installation because normal site-packages is not writeable
Collecting flask
  Using cached flask-3.0.3-py3-none-any.whl (101 kB)
Collecting blinker>=1.6.2
  Using cached blinker-1.7.0-py3-none-any.whl (13 kB)
Collecting Werkzeug>=3.0.0
  Using cached werkzeug-3.0.2-py3-none-any.whl (226 kB)
Collecting click>=8.1.3
  Using cached click-8.1.7-py3-none-any.whl (97 kB)
Collecting itsdangerous>=2.1.2
  Using cached itsdangerous-2.2.0-py3-none-any.whl (16 kB)
Requirement already satisfied: Jinja2>=3.1.2 in /home/openhands/.local/lib/python3.10/site-packages (from flask) (3.1.3)
Requirement already satisfied: MarkupSafe>=2.0 in /home/openhands/.local/lib/python3.10/site-packages (from Jinja2>=3.1.2->flask) (2.1.5)
Installing collected packages: Werkzeug, itsdangerous, click, blinker, flask
Successfully installed Werkzeug-3.0.2 blinker-1.7.0 click-8.1.7 flask-3.0.3 itsdangerous-2.2.0

ASSISTANT:
The `flask` module is installed successfully. Let me run the Python file again:
<function=terminal>
<parameter=command>
python3 app.py > server.log 2>&1 &
</parameter>
</function>

USER: EXECUTION RESULT of [terminal]:
[1] 124

ASSISTANT:
Let me check the server log again:
<function=terminal>
<parameter=command>
cat server.log
</parameter>
</function>

USER: EXECUTION RESULT of [terminal]:
* Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit

ASSISTANT:
The server is running on port 5000 with PID 124. You can access the list of numbers by visiting http://127.0.0.1:5000. If you have any further questions, feel free to ask!
""",  # noqa: E501
        "kill_server": """
USER: Now kill the server, make it display the numbers in a table format.

ASSISTANT:
Sure! Let me stop the server first:
<function=terminal>
<parameter=command>
kill 124
</parameter>
</function>

USER: EXECUTION RESULT of [terminal]:
[1]+  Terminated              python3 app.py > server.log 2>&1
""",  # noqa: E501
        "run_server_again": """
ASSISTANT:
Running the updated file:
<function=terminal>
<parameter=command>
python3 app.py > server.log 2>&1 &
</parameter>
</function>

USER: EXECUTION RESULT of [terminal]:
[1] 126

ASSISTANT:
The server is running on port 5000 with PID 126. You can access the list of numbers in a table format by visiting http://127.0.0.1:5000.
""",  # noqa: E501
    },
    "file_editor": {
        "create_file": """
ASSISTANT:
There is no `app.py` file in the current directory. Let me create a Python file `app.py`:
<function=str_replace_editor>
<parameter=command>create</parameter>
<parameter=path>/workspace/app.py</parameter>
<parameter=file_text>
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    numbers = list(range(1, 11))
    return str(numbers)

if __name__ == '__main__':
    app.run(port=5000)
</parameter>
</function>

USER: EXECUTION RESULT of [str_replace_editor]:
File created successfully at: /workspace/app.py
""",  # noqa: E501
        "edit_file": """
ASSISTANT:
Now let me display the numbers in a table format:
<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/workspace/app.py</parameter>
<parameter=old_str>return str(numbers)</parameter>
<parameter=new_str>return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'</parameter>
</function>

USER: EXECUTION RESULT of [str_replace_editor]:
The file /workspace/app.py has been edited. Here's the result of running `cat -n` on a snippet of /workspace/app.py:
     3
     4  @app.route('/')
     5  def index():
     6      numbers = list(range(1, 11))
     7      return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'
     8
     9  if __name__ == '__main__':
    10      app.run(port=5000)
Review the changes and make sure they are as expected. Edit the file again if necessary.

""",  # noqa: E501
    },
    "browser": {
        "view_page": """
ASSISTANT:
Let me check how the page looks in the browser:
<function=browser>
<parameter=code>
goto('http://127.0.0.1:5000')
noop(1000)  # Wait for page to load
</parameter>
</function>

USER: EXECUTION RESULT of [browser]:
[Browser shows the numbers in a table format]
"""  # noqa: E501
    },
    "edit_file": {
        "create_file": """
ASSISTANT: There is no `app.py` file in the current directory. Let me create a Python file `app.py`:
<function=edit_file>
<parameter=path>/workspace/app.py</parameter>
<parameter=start>1</parameter>
<parameter=end>-1</parameter>
<parameter=content>
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    numbers = list(range(1, 11))
    return str(numbers)

if __name__ == '__main__':
    app.run(port=5000)
</parameter>
</function>

USER: EXECUTION RESULT of [edit_file]:
File created successfully at: /workspace/app.py
""",  # noqa: E501
        "edit_file": """
ASSISTANT:
Now let me display the numbers in a table format:
<function=edit_file>
<parameter=path>/workspace/app.py</parameter>
<parameter=start>6</parameter>
<parameter=end>9</parameter>
<parameter=content>
    numbers = list(range(1, 11))
    return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'
    # ... existing code ...
if __name__ == '__main__':
</parameter>
</function>

USER: EXECUTION RESULT of [edit_file]:
The file /workspace/app.py has been edited. Here's the result of running `cat -n` on a snippet of /workspace/app.py:
     3
     4  @app.route('/')
     5  def index():
     6      numbers = list(range(1, 11))
     7      return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'
     8
     9  if __name__ == '__main__':
    10      app.run(port=5000)
Review the changes and make sure they are as expected. Edit the file again if necessary.
""",  # noqa: E501
    },
    "finish": {
        "example": """
ASSISTANT:
The server is running on port 5000 with PID 126. You can access the list of numbers in a table format by visiting http://127.0.0.1:5000. Let me know if you have any further requests!
<function=finish>
<parameter=message>The task has been completed. The web server is running and displaying numbers 1-10 in a table format at http://127.0.0.1:5000.</parameter>
</function>
"""  # noqa: E501
    },
    "task_tracker": {
        "view": """
ASSISTANT:
Let me check the current task list first:
<function=task_tracker>
<parameter=command>view</parameter>
</function>
""",
        "plan": """
I'll create or update the full plan based on your requirements and current progress:
<function=task_tracker>
<parameter=command>plan</parameter>
<parameter=task_list>
[
  {
    "title": "Initialize repo",
    "status": "done",
    "notes": "Repository created and README added."
  },
  {
    "title": "Implement nested param parsing",
    "status": "in_progress",
    "notes": "Add recursive parsing for array-typed parameters."
  }
]
</parameter>
</function>
""",
    },
}


def get_example_for_tools(tools: list[ChatCompletionToolParam]) -> str:
    """Generate an in-context learning example based on available tools."""
    available_tools = set()
    for tool in tools:
        if tool["type"] == "function":
            name = tool["function"]["name"]
            if name == TERMINAL_TOOL_NAME:
                available_tools.add("terminal")
            elif name == STR_REPLACE_EDITOR_TOOL_NAME:
                available_tools.add("file_editor")
            elif name == BROWSER_TOOL_NAME:
                available_tools.add("browser")
            elif name == FINISH_TOOL_NAME:
                available_tools.add("finish")
            elif name == LLM_BASED_EDIT_TOOL_NAME:
                available_tools.add("edit_file")
            elif name == TASK_TRACKER_TOOL_NAME:
                available_tools.add("task_tracker")

    if not available_tools:
        return ""

    example = """Here's a running example of how to perform a task with the provided tools.

--------------------- START OF EXAMPLE ---------------------

USER: Create a list of numbers from 1 to 10, and display them in a web page at port 5000.

"""  # noqa: E501

    # Build example based on available tools
    if "terminal" in available_tools:
        example += TOOL_EXAMPLES["bash"]["check_dir"]

    if "file_editor" in available_tools:
        example += TOOL_EXAMPLES["file_editor"]["create_file"]
    elif "edit_file" in available_tools:
        example += TOOL_EXAMPLES["edit_file"]["create_file"]

    if "terminal" in available_tools:
        example += TOOL_EXAMPLES["bash"]["run_server"]

    if "browser" in available_tools:
        example += TOOL_EXAMPLES["browser"]["view_page"]

    if "terminal" in available_tools:
        example += TOOL_EXAMPLES["bash"]["kill_server"]

    if "file_editor" in available_tools:
        example += TOOL_EXAMPLES["file_editor"]["edit_file"]
    elif "edit_file" in available_tools:
        example += TOOL_EXAMPLES["edit_file"]["edit_file"]

    if "terminal" in available_tools:
        example += TOOL_EXAMPLES["bash"]["run_server_again"]

    if "finish" in available_tools:
        example += TOOL_EXAMPLES["finish"]["example"]

    if "task_tracker" in available_tools:
        example += TOOL_EXAMPLES["task_tracker"]["view"]
        example += TOOL_EXAMPLES["task_tracker"]["plan"]

    example += """
--------------------- END OF EXAMPLE ---------------------

Do NOT assume the environment is the same as in the example above.

--------------------- NEW TASK DESCRIPTION ---------------------
"""  # noqa: E501
    example = example.lstrip()

    return refine_prompt(example)


IN_CONTEXT_LEARNING_EXAMPLE_PREFIX = get_example_for_tools

IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX = """
--------------------- END OF NEW TASK DESCRIPTION ---------------------

PLEASE follow the format strictly! PLEASE EMIT ONE AND ONLY ONE FUNCTION CALL PER MESSAGE.
"""  # noqa: E501

# Regex patterns for function call parsing
# Note: newline after function name is optional for compatibility with various models
FN_REGEX_PATTERN = r"<function=([^>]+)>\n?(.*?)</function>"
FN_PARAM_REGEX_PATTERN = r"<parameter=([^>]+)>(.*?)</parameter>"

# Add new regex pattern for tool execution results
TOOL_RESULT_REGEX_PATTERN = r"EXECUTION RESULT of \[(.*?)\]:\n(.*)"


def convert_tool_call_to_string(tool_call: dict) -> str:
    """Convert tool call to content in string format."""
    if "function" not in tool_call:
        raise FunctionCallConversionError("Tool call must contain 'function' key.")
    if "id" not in tool_call:
        raise FunctionCallConversionError("Tool call must contain 'id' key.")
    if "type" not in tool_call:
        raise FunctionCallConversionError("Tool call must contain 'type' key.")
    if tool_call["type"] != "function":
        raise FunctionCallConversionError("Tool call type must be 'function'.")

    ret = f"<function={tool_call['function']['name']}>\n"
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError as e:
        raise FunctionCallConversionError(
            f"Failed to parse arguments as JSON. "
            f"Arguments: {tool_call['function']['arguments']}"
        ) from e
    for param_name, param_value in args.items():
        is_multiline = isinstance(param_value, str) and "\n" in param_value
        ret += f"<parameter={param_name}>"
        if is_multiline:
            ret += "\n"
        if isinstance(param_value, list) or isinstance(param_value, dict):
            ret += json.dumps(param_value)
        else:
            ret += f"{param_value}"
        if is_multiline:
            ret += "\n"
        ret += "</parameter>\n"
    ret += "</function>"
    return ret


def _summarize_schema_type(schema: object | None) -> str:
    """
    Capture array, union, enum, and nested type info.
    """
    if not isinstance(schema, dict):
        return "unknown" if schema is None else str(schema)

    for key in SCHEMA_UNION_KEYS:
        if key in schema:
            return " or ".join(_summarize_schema_type(option) for option in schema[key])

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " or ".join(str(t) for t in schema_type)
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, list):
            item_types = ", ".join(_summarize_schema_type(item) for item in items)
            return f"array[{item_types}]"
        if isinstance(items, dict):
            return f"array[{_summarize_schema_type(items)}]"
        return "array"
    if schema_type:
        return str(schema_type)
    if "enum" in schema:
        return "enum"
    return "unknown"


def _indent(indent: int) -> str:
    return " " * indent


def _nested_indent(indent: int, levels: int = 1) -> int:
    return indent + SCHEMA_INDENT_STEP * levels


def _get_description(schema: dict[str, object] | None) -> str:
    """
    Extract description from schema, or return placeholder if missing.
    """
    if not isinstance(schema, dict):
        return MISSING_DESCRIPTION_PLACEHOLDER
    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        return description
    return MISSING_DESCRIPTION_PLACEHOLDER


def _format_union_details(schema: dict[str, object], indent: int) -> list[str] | None:
    for key in SCHEMA_UNION_KEYS:
        options = schema.get(key)
        if not isinstance(options, list):
            continue
        lines = [f"{_indent(indent)}{key} options:"]
        for option in options:
            option_type = _summarize_schema_type(option)
            option_line = f"{_indent(_nested_indent(indent))}- {option_type}"
            option_line += (
                f": {_get_description(option if isinstance(option, dict) else None)}"
            )
            lines.append(option_line)
            lines.extend(_format_schema_detail(option, _nested_indent(indent, 2)))
        return lines
    return None


def _format_array_details(schema: dict[str, object], indent: int) -> list[str]:
    lines = [f"{_indent(indent)}Array items:"]
    items = schema.get("items")
    if isinstance(items, list):
        for index, item_schema in enumerate(items):
            item_type = _summarize_schema_type(item_schema)
            lines.append(
                f"{_indent(_nested_indent(indent))}- index {index}: {item_type}"
            )
            lines.extend(_format_schema_detail(item_schema, _nested_indent(indent, 2)))
    elif isinstance(items, dict):
        lines.append(
            f"{_indent(_nested_indent(indent))}Type: {_summarize_schema_type(items)}"
        )
        lines.extend(_format_schema_detail(items, _nested_indent(indent, 2)))
    else:
        lines.append(f"{_indent(_nested_indent(indent))}Type: unknown")
    return lines


def _format_additional_properties(
    additional_props: object | None, indent: int
) -> list[str]:
    if isinstance(additional_props, dict):
        line = (
            f"{_indent(indent)}Additional properties allowed: "
            f"{_summarize_schema_type(additional_props)}"
        )
        lines = [line]
        lines.extend(_format_schema_detail(additional_props, _nested_indent(indent)))
        return lines
    if additional_props is True:
        return [f"{_indent(indent)}Additional properties allowed."]
    if additional_props is False:
        return [f"{_indent(indent)}Additional properties not allowed."]
    return []


def _format_object_details(schema: dict[str, Any], indent: int) -> list[str]:
    lines: list[str] = []
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    if isinstance(properties, dict) and properties:
        lines.append(f"{_indent(indent)}Object properties:")
        for name, prop in properties.items():
            prop_type = _summarize_schema_type(prop)
            required_flag = "required" if name in required else "optional"
            prop_desc = _get_description(prop if isinstance(prop, dict) else None)
            lines.append(
                f"{_indent(_nested_indent(indent))}- {name} ({prop_type},"
                f" {required_flag}): {prop_desc}"
            )
            lines.extend(_format_schema_detail(prop, _nested_indent(indent, 2)))
    lines.extend(
        _format_additional_properties(schema.get("additionalProperties"), indent)
    )
    return lines


def _format_schema_detail(schema: object | None, indent: int = 4) -> list[str]:
    """Recursively describe arrays, objects, unions, and additional properties."""
    if not isinstance(schema, dict):
        return []

    union_lines = _format_union_details(schema, indent)
    if union_lines is not None:
        return union_lines

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        allowed_types = ", ".join(str(t) for t in schema_type)
        return [f"{_indent(indent)}Allowed types: {allowed_types}"]

    if schema_type == "array":
        return _format_array_details(schema, indent)

    if schema_type == "object":
        return _format_object_details(schema, indent)

    return []


def convert_tools_to_description(tools: list[ChatCompletionToolParam]) -> str:
    ret = ""
    for i, tool in enumerate(tools):
        assert tool["type"] == "function"
        fn = tool["function"]
        if i > 0:
            ret += "\n"
        ret += f"---- BEGIN FUNCTION #{i + 1}: {fn['name']} ----\n"
        if "description" in fn:
            ret += f"Description: {fn['description']}\n"

        if "parameters" in fn:
            ret += "Parameters:\n"
            properties = fn["parameters"].get("properties", {})
            required_params = set(fn["parameters"].get("required", []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                is_required = param_name in required_params
                param_status = "required" if is_required else "optional"
                param_type = _summarize_schema_type(param_info)

                desc = _get_description(
                    param_info if isinstance(param_info, dict) else None
                )

                if "enum" in param_info:
                    enum_values = ", ".join(f"`{v}`" for v in param_info["enum"])
                    desc += f"\nAllowed values: [{enum_values}]"

                ret += (
                    f"  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n"
                )

                detail_lines = _format_schema_detail(param_info, indent=6)
                if detail_lines:
                    ret += "\n".join(detail_lines) + "\n"

        else:
            ret += "No parameters are required for this function.\n"

        ret += f"---- END FUNCTION #{i + 1} ----\n"
    return ret


def convert_fncall_messages_to_non_fncall_messages(
    messages: list[dict],
    tools: list[ChatCompletionToolParam],
    add_in_context_learning_example: bool = True,
) -> list[dict]:
    """Convert function calling messages to non-function calling messages."""
    messages = copy.deepcopy(messages)

    formatted_tools = convert_tools_to_description(tools)
    system_message_suffix = system_message_suffix_TEMPLATE.format(
        description=formatted_tools
    )

    converted_messages = []
    first_user_message_encountered = False
    for message in messages:
        role = message["role"]
        content: Content = message.get("content") or ""

        # 1. SYSTEM MESSAGES
        # append system prompt suffix to content
        if role == "system":
            if isinstance(content, str):
                content += system_message_suffix
            elif isinstance(content, list):
                if content and content[-1]["type"] == "text":
                    content[-1]["text"] += system_message_suffix
                else:
                    content.append({"type": "text", "text": system_message_suffix})
            else:
                raise FunctionCallConversionError(
                    f"Unexpected content type {type(content)}. "
                    f"Expected str or list. "
                    f"Content: {content}"
                )
            converted_messages.append({"role": "system", "content": content})

        # 2. USER MESSAGES (no change)
        elif role == "user":
            # Add in-context learning example for the first user message
            if not first_user_message_encountered and add_in_context_learning_example:
                first_user_message_encountered = True

                # Generate example based on available tools
                example = IN_CONTEXT_LEARNING_EXAMPLE_PREFIX(tools)

                # Add example if we have any tools
                if example:
                    # add in-context learning example
                    if isinstance(content, str):
                        content = example + content + IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX
                    elif isinstance(content, list):
                        if content and content[0]["type"] == "text":
                            content[0]["text"] = (
                                example
                                + content[0]["text"]
                                + IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX
                            )
                        else:
                            content = (
                                [
                                    cast(
                                        TextPart,
                                        {
                                            "type": "text",
                                            "text": example,
                                        },
                                    )
                                ]
                                + content
                                + [
                                    cast(
                                        TextPart,
                                        {
                                            "type": "text",
                                            "text": IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX,
                                        },
                                    )
                                ]
                            )
                    else:
                        raise FunctionCallConversionError(
                            f"Unexpected content type {type(content)}. "
                            f"Expected str or list. "
                            f"Content: {content}"
                        )
            converted_messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )

        # 3. ASSISTANT MESSAGES
        # - 3.1 no change if no function call
        # - 3.2 change if function call
        elif role == "assistant":
            if "tool_calls" in message and message["tool_calls"] is not None:
                if len(message["tool_calls"]) != 1:
                    raise FunctionCallConversionError(
                        f"Expected exactly one tool call in the message. "
                        f"More than one tool call is not supported. "
                        f"But got {len(message['tool_calls'])} tool calls. "
                        f"Content: {content}"
                    )
                try:
                    tool_content = convert_tool_call_to_string(message["tool_calls"][0])
                except FunctionCallConversionError as e:
                    raise FunctionCallConversionError(
                        f"Failed to convert tool call to string.\n"
                        f"Current tool call: {message['tool_calls'][0]}.\n"
                        f"Raw messages: {json.dumps(messages, indent=2)}"
                    ) from e
                if isinstance(content, str):
                    content += "\n\n" + tool_content
                    content = content.lstrip()
                elif isinstance(content, list):
                    if content and content[-1]["type"] == "text":
                        content[-1]["text"] += "\n\n" + tool_content
                        content[-1]["text"] = content[-1]["text"].lstrip()
                    else:
                        content.append({"type": "text", "text": tool_content})
                else:
                    raise FunctionCallConversionError(
                        f"Unexpected content type {type(content)}. "
                        f"Expected str or list. Content: {content}"
                    )
            converted_messages.append({"role": "assistant", "content": content})

        # 4. TOOL MESSAGES (tool outputs)
        elif role == "tool":
            # Convert tool result as user message
            tool_name = message.get("name", "function")
            prefix = f"EXECUTION RESULT of [{tool_name}]:\n"
            # and omit "tool_call_id" AND "name"
            if isinstance(content, str):
                content = prefix + content
            elif isinstance(content, list):
                if content and (
                    first_text_content := next(
                        (c for c in content if c["type"] == "text"), None
                    )
                ):
                    first_text_content["text"] = prefix + first_text_content["text"]
                else:
                    content = [
                        cast(TextPart, {"type": "text", "text": prefix})
                    ] + content

                if "cache_control" in message:
                    content[-1]["cache_control"] = cast(
                        CacheControl, {"type": "ephemeral"}
                    )
            else:
                raise FunctionCallConversionError(
                    f"Unexpected content type {type(content)}. "
                    f"Expected str or list. "
                    f"Content: {content}"
                )

            converted_messages.append({"role": "user", "content": content})
        else:
            raise FunctionCallConversionError(
                f"Unexpected role {role}. Expected system, user, assistant or tool."
            )
    return converted_messages


def _extract_and_validate_params(
    matching_tool: ChatCompletionToolParamFunctionChunk,
    param_matches: Iterable[re.Match],
    fn_name: str,
) -> dict:
    params = {}
    # Parse and validate parameters
    required_params = set()
    if "parameters" in matching_tool and "required" in matching_tool["parameters"]:
        required_params = set(matching_tool["parameters"].get("required", []))

    allowed_params = set()
    if "parameters" in matching_tool and "properties" in matching_tool["parameters"]:
        allowed_params = set(matching_tool["parameters"]["properties"].keys())

    param_name_to_type = {}
    if "parameters" in matching_tool and "properties" in matching_tool["parameters"]:
        param_name_to_type = {
            name: val.get("type", "string")
            for name, val in matching_tool["parameters"]["properties"].items()
        }

    # Collect parameters
    found_params = set()
    for param_match in param_matches:
        param_name = param_match.group(1)
        param_value = param_match.group(2)
        # Normalize whitespace: some models add extra newlines around values
        if isinstance(param_value, str):
            param_value = param_value.strip()

        # Validate parameter is allowed
        if allowed_params and param_name not in allowed_params:
            raise FunctionCallValidationError(
                f"Parameter '{param_name}' is not allowed for function '{fn_name}'. "
                f"Allowed parameters: {allowed_params}"
            )

        # Validate and convert parameter type
        # supported: string, integer, array
        if param_name in param_name_to_type:
            if param_name_to_type[param_name] == "integer":
                try:
                    param_value = int(param_value)
                except ValueError:
                    raise FunctionCallValidationError(
                        f"Parameter '{param_name}' is expected to be an integer."
                    )
            elif param_name_to_type[param_name] == "array":
                try:
                    param_value = json.loads(param_value)
                except json.JSONDecodeError:
                    raise FunctionCallValidationError(
                        f"Parameter '{param_name}' is expected to be an array."
                    )
            else:
                # string
                pass

        # Enum check
        if (
            "parameters" in matching_tool
            and "enum" in matching_tool["parameters"]["properties"][param_name]
        ):
            if (
                param_value
                not in matching_tool["parameters"]["properties"][param_name]["enum"]
            ):
                raise FunctionCallValidationError(
                    f"Parameter '{param_name}' is expected to be one of "
                    f"{matching_tool['parameters']['properties'][param_name]['enum']}."
                )

        params[param_name] = param_value
        found_params.add(param_name)

    # Check all required parameters are present
    # Note: security_risk is excluded here because its validation happens later
    # in Agent._extract_security_risk(), which has context about whether a security
    # analyzer is configured. This allows weaker models to omit it when no analyzer
    # is active, while still enforcing it for stronger models with LLMSecurityAnalyzer.
    missing_params = required_params - found_params - {"security_risk"}
    if missing_params:
        raise FunctionCallValidationError(
            f"Missing required parameters for function '{fn_name}': {missing_params}"
        )
    return params


def _preprocess_model_output(content: str) -> str:
    """Clean up model-specific formatting before parsing function calls.

    Removes wrapper tags that some models (like Nemotron) emit around function calls:
    - </think> before the function call
    - <tool_call>...</tool_call> around the function call

    Only strips tags at boundaries, not inside parameter values.
    """
    # Strip </think> when it appears before <function= (Nemotron reasoning end)
    content = re.sub(r"</think>\s*(?=<function=)", "", content)
    # Strip <tool_call> when it appears right before <function=
    content = re.sub(r"<tool_call>\s*(?=<function=)", "", content)
    # Strip </tool_call> when it appears right after </function>
    content = re.sub(r"(?<=</function>)\s*</tool_call>", "", content)
    return content


def _fix_stopword(content: str) -> str:
    """Fix the issue when some LLM would NOT return the stopword."""
    content = _preprocess_model_output(content)
    if "<function=" in content and content.count("<function=") == 1:
        if content.endswith("</"):
            content = content.rstrip() + "function>"
        elif not content.rstrip().endswith("</function>"):
            content = content + "\n</function>"
    return content


def _normalize_parameter_tags(fn_body: str) -> str:
    """Normalize malformed parameter tags to the canonical format.

    Some models occasionally emit malformed parameter tags like:
        <parameter=command=str_replace</parameter>
    instead of the correct:
        <parameter=command>str_replace</parameter>

    This function rewrites the malformed form into the correct one to allow
    downstream parsing to succeed.
    """
    # Replace '<parameter=name=value</parameter>'
    # with '<parameter=name>value</parameter>'
    return re.sub(
        r"<parameter=([a-zA-Z0-9_]+)=([^<]*)</parameter>",
        r"<parameter=\1>\2</parameter>",
        fn_body,
    )


def convert_non_fncall_messages_to_fncall_messages(
    messages: list[dict],
    tools: list[ChatCompletionToolParam],
) -> list[dict]:
    """Convert non-function calling messages back to function calling messages."""
    messages = copy.deepcopy(messages)
    formatted_tools = convert_tools_to_description(tools)
    system_message_suffix = system_message_suffix_TEMPLATE.format(
        description=formatted_tools
    )

    converted_messages = []
    tool_call_counter = 1  # Counter for tool calls

    first_user_message_encountered = False
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        # For system messages, remove the added suffix
        if role == "system":
            if isinstance(content, str):
                # Remove the suffix if present
                content = content.split(system_message_suffix)[0]
            elif isinstance(content, list):
                if content and content[-1]["type"] == "text":
                    # Remove the suffix from the last text item
                    content[-1]["text"] = content[-1]["text"].split(
                        system_message_suffix
                    )[0]
            converted_messages.append({"role": "system", "content": content})
        # Skip user messages (no conversion needed)
        elif role == "user":
            # Check & replace in-context learning example
            if not first_user_message_encountered:
                first_user_message_encountered = True
                if isinstance(content, str):
                    # Remove any existing example
                    if content.startswith(IN_CONTEXT_LEARNING_EXAMPLE_PREFIX(tools)):
                        content = content.replace(
                            IN_CONTEXT_LEARNING_EXAMPLE_PREFIX(tools), "", 1
                        )
                    if content.endswith(IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX):
                        content = content.replace(
                            IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX, "", 1
                        )
                elif isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            # Remove any existing example
                            example = IN_CONTEXT_LEARNING_EXAMPLE_PREFIX(tools)
                            if item["text"].startswith(example):
                                item["text"] = item["text"].replace(example, "", 1)
                            if item["text"].endswith(
                                IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX
                            ):
                                item["text"] = item["text"].replace(
                                    IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX, "", 1
                                )
                else:
                    raise FunctionCallConversionError(
                        f"Unexpected content type {type(content)}. "
                        f"Expected str or list. "
                        f"Content: {content}"
                    )

            # Check for tool execution result pattern
            if isinstance(content, str):
                tool_result_match = re.search(
                    TOOL_RESULT_REGEX_PATTERN, content, re.DOTALL
                )
            elif isinstance(content, list):
                tool_result_match = next(
                    (
                        _match
                        for item in content
                        if item.get("type") == "text"
                        and (
                            _match := re.search(
                                TOOL_RESULT_REGEX_PATTERN, item["text"], re.DOTALL
                            )
                        )
                    ),
                    None,
                )
            else:
                raise FunctionCallConversionError(
                    f"Unexpected content type {type(content)}. "
                    f"Expected str or list. "
                    f"Content: {content}"
                )

            if tool_result_match:
                if isinstance(content, list):
                    text_content_items = [
                        item for item in content if item.get("type") == "text"
                    ]
                    if not text_content_items:
                        raise FunctionCallConversionError(
                            f"Could not find text content in message with tool result. "
                            f"Content: {content}"
                        )
                elif not isinstance(content, str):
                    raise FunctionCallConversionError(
                        f"Unexpected content type {type(content)}. "
                        f"Expected str or list. "
                        f"Content: {content}"
                    )

                tool_name = tool_result_match.group(1)
                tool_result = tool_result_match.group(2).strip()

                # Convert to tool message format
                converted_messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": [{"type": "text", "text": tool_result}]
                        if isinstance(content, list)
                        else tool_result,
                        "tool_call_id": f"toolu_{tool_call_counter - 1:02d}",
                        # Use last generated ID
                    }
                )
            else:
                converted_messages.append({"role": "user", "content": content})

        # Handle assistant messages
        elif role == "assistant":
            if isinstance(content, str):
                content = _fix_stopword(content)
                fn_match = re.search(FN_REGEX_PATTERN, content, re.DOTALL)
            elif isinstance(content, list):
                if content and content[-1]["type"] == "text":
                    content[-1]["text"] = _fix_stopword(content[-1]["text"])
                    fn_match = re.search(
                        FN_REGEX_PATTERN, content[-1]["text"], re.DOTALL
                    )
                else:
                    fn_match = None
                fn_match_exists = any(
                    item.get("type") == "text"
                    and re.search(FN_REGEX_PATTERN, item["text"], re.DOTALL)
                    for item in content
                )
                if fn_match_exists and not fn_match:
                    raise FunctionCallConversionError(
                        f"Expecting function call in the LAST index of content list. "
                        f"But got content={content}"
                    )
            else:
                raise FunctionCallConversionError(
                    f"Unexpected content type {type(content)}. "
                    f"Expected str or list. "
                    f"Content: {content}"
                )

            if fn_match:
                fn_name = fn_match.group(1)
                fn_body = _normalize_parameter_tags(fn_match.group(2))

                def _find_tool(
                    name: str,
                ) -> ChatCompletionToolParamFunctionChunk | None:
                    return next(
                        (
                            tool["function"]
                            for tool in tools
                            if tool["type"] == "function"
                            and tool["function"]["name"] == name
                        ),
                        None,
                    )

                matching_tool = _find_tool(fn_name)
                # Try aliases if tool not found (some models use legacy names)
                if not matching_tool:
                    TOOL_NAME_ALIASES = {
                        "str_replace_editor": "file_editor",
                        "bash": "terminal",
                        "execute_bash": "terminal",
                        "str_replace": "file_editor",
                    }
                    if fn_name in TOOL_NAME_ALIASES:
                        fn_name = TOOL_NAME_ALIASES[fn_name]
                        matching_tool = _find_tool(fn_name)
                # Validate function exists in tools
                if not matching_tool:
                    available_tools = [
                        tool["function"]["name"]
                        for tool in tools
                        if tool["type"] == "function"
                    ]
                    raise FunctionCallValidationError(
                        f"Function '{fn_name}' not found in available tools: "
                        f"{available_tools}"
                    )

                # Parse parameters
                param_matches = re.finditer(FN_PARAM_REGEX_PATTERN, fn_body, re.DOTALL)
                params = _extract_and_validate_params(
                    matching_tool, param_matches, fn_name
                )

                # Create tool call with unique ID
                tool_call_id = f"toolu_{tool_call_counter:02d}"
                tool_call = {
                    "index": 1,  # always 1 because we only support
                    # **one tool call per message**
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": fn_name, "arguments": json.dumps(params)},
                }
                tool_call_counter += 1  # Increment counter

                # Remove the function call part from content
                if isinstance(content, list):
                    assert content and content[-1]["type"] == "text"
                    content[-1]["text"] = (
                        content[-1]["text"].split("<function=")[0].strip()
                    )
                elif isinstance(content, str):
                    content = content.split("<function=")[0].strip()
                else:
                    raise FunctionCallConversionError(
                        f"Unexpected content type {type(content)}. "
                        f"Expected str or list. "
                        f"Content: {content}"
                    )

                converted_messages.append(
                    {"role": "assistant", "content": content, "tool_calls": [tool_call]}
                )
            else:
                # No function call, keep message as is
                converted_messages.append(message)

        else:
            raise FunctionCallConversionError(
                f"Unexpected role {role}. Expected system, user, or assistant "
                f"in non-function calling messages."
            )
    return converted_messages


def convert_from_multiple_tool_calls_to_single_tool_call_messages(
    messages: list[dict],
    ignore_final_tool_result: bool = False,
) -> list[dict]:
    """Break one message with multiple tool calls into multiple messages."""
    converted_messages = []

    pending_tool_calls: dict[str, dict] = {}
    for message in messages:
        role: str
        content: Content
        role = message["role"]
        content = message.get("content") or ""
        if role == "assistant":
            if message.get("tool_calls") and len(message["tool_calls"]) > 1:
                # handle multiple tool calls by breaking them into multiple messages
                for i, tool_call in enumerate(message["tool_calls"]):
                    pending_tool_calls[tool_call["id"]] = {
                        "role": "assistant",
                        "content": content if i == 0 else "",
                        "tool_calls": [tool_call],
                    }
            else:
                converted_messages.append(message)
        elif role == "tool":
            if message["tool_call_id"] in pending_tool_calls:
                # remove the tool call from the pending list
                _tool_call_message = pending_tool_calls.pop(message["tool_call_id"])
                converted_messages.append(_tool_call_message)
                # add the tool result
                converted_messages.append(message)
            else:
                assert len(pending_tool_calls) == 0, (
                    f"Found pending tool calls but not found in pending list: "
                    f"{pending_tool_calls=}"
                )
                converted_messages.append(message)
        else:
            assert len(pending_tool_calls) == 0, (
                f"Found pending tool calls but not expect to handle it "
                f"with role {role}: "
                f"{pending_tool_calls=}, {message=}"
            )
            converted_messages.append(message)

    if not ignore_final_tool_result and len(pending_tool_calls) > 0:
        raise FunctionCallConversionError(
            f"Found pending tool calls but no tool result: {pending_tool_calls=}"
        )
    return converted_messages
