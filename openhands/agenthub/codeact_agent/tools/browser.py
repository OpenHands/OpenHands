# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from browsergym.core.action.highlevel import HighLevelActionSet
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.agenthub.codeact_agent.tools.security_utils import (
    RISK_LEVELS,
    SECURITY_RISK_DESC,
)
from openhands.llm.tool_names import BROWSER_TOOL_NAME

# from browsergym/core/action/highlevel.py
_browser_action_space = HighLevelActionSet(
    subsets=['bid', 'nav'],
    strict=False,  # less strict on the parsing of the actions
    multiaction=True,  # enable to agent to take multiple actions at once
)


_BROWSER_DESCRIPTION = """Interact with the browser using Python code. Use it ONLY when you need to interact with a webpage.

See the description of "code" parameter for more details.

Multiple actions can be provided at once, but will be executed sequentially without any feedback from the page.
More than 2-3 actions usually leads to failure or unexpected behavior. Example:
fill('a12', 'example with "quotes"')
click('a51')
click('48', button='middle', modifiers=['Shift'])

You can also use the browser to view pdf, png, jpg files.
You should first check the content of /tmp/oh-server-url to get the server url, and then use it to view the file by `goto("{server_url}/view?path={absolute_file_path}")`.
For example: `goto("http://localhost:8000/view?path=/workspace/test_document.pdf")`
Note: The file should be downloaded to the local machine first before using the browser to view it.
"""

_BROWSER_TOOL_DESCRIPTION = """
The following 15 functions are available. Nothing else is supported.

goto(url: str)
    Description: Navigate to a url.
    Examples:
        goto('http://www.example.com')

go_back()
    Description: Navigate to the previous page in history.
    Examples:
        go_back()

go_forward()
    Description: Navigate to the next page in history.
    Examples:
        go_forward()

noop(wait_ms: float = 1000)
    Description: Do nothing, and optionally wait for the given time (in milliseconds).
    You can use this to get the current page content and/or wait for the page to load.
    Examples:
        noop()

        noop(500)

scroll(delta_x: float, delta_y: float)
    Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.
    Examples:
        scroll(0, 200)

        scroll(-50.2, -100.5)

fill(bid: str, value: str)
    Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.
    Examples:
        fill('237', 'example value')

        fill('45', 'multi-line\nexample')

        fill('a12', 'example with "quotes"')

select_option(bid: str, options: str | list[str])
    Description: Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.
    Examples:
        select_option('a48', 'blue')

        select_option('c48', ['red', 'green', 'blue'])

click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
    Description: Click an element.
    Examples:
        click('a51')

        click('b22', button='right')

        click('48', button='middle', modifiers=['Shift'])

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
    Description: Double click an element.
    Examples:
        dblclick('12')

        dblclick('ca42', button='right')

        dblclick('178', button='middle', modifiers=['Shift'])

hover(bid: str)
    Description: Hover over an element.
    Examples:
        hover('b8')

press(bid: str, key_comb: str)
    Description: Focus the matching element and press a combination of keys. It accepts the logical key names that are emitted in the keyboardEvent.key property of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDown, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can alternatively specify a single character you'd like to produce such as "a" or "#". Following modification shortcuts are also supported: Shift, Control, Alt, Meta, ShiftLeft, ControlOrMeta. ControlOrMeta resolves to Control on Windows and Linux and to Meta on macOS.
    Examples:
        press('88', 'Backspace')

        press('a26', 'ControlOrMeta+a')

        press('a61', 'Meta+Shift+t')

focus(bid: str)
    Description: Focus the matching element.
    Examples:
        focus('b455')

clear(bid: str)
    Description: Clear the input field.
    Examples:
        clear('996')

drag_and_drop(from_bid: str, to_bid: str)
    Description: Perform a drag & drop. Hover the element that will be dragged. Press left mouse button. Move mouse to the element that will receive the drop. Release left mouse button.
    Examples:
        drag_and_drop('56', '498')

upload_file(bid: str, file: str | list[str])
    Description: Click an element and wait for a "filechooser" event, then select one or multiple input files for upload. Relative file paths are resolved relative to the current working directory. An empty list clears the selected files.
    Examples:
        upload_file('572', '/home/user/my_receipt.pdf')

        upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])
"""


for _, action in _browser_action_space.action_set.items():
    assert action.signature in _BROWSER_TOOL_DESCRIPTION, (
        f'Browser description mismatch. Please double check if the BrowserGym updated their action space.\n\nAction: {action.signature}'
    )
    assert action.description in _BROWSER_TOOL_DESCRIPTION, (
        f'Browser description mismatch. Please double check if the BrowserGym updated their action space.\n\nAction: {action.description}'
    )

BrowserTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=BROWSER_TOOL_NAME,
        description=_BROWSER_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': (
                        'The Python code that interacts with the browser.\n'
                        + _BROWSER_TOOL_DESCRIPTION
                    ),
                },
                'security_risk': {
                    'type': 'string',
                    'description': SECURITY_RISK_DESC,
                    'enum': RISK_LEVELS,
                },
            },
            'required': ['code', 'security_risk'],
        },
    ),
)
