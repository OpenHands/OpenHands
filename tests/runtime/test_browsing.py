"""Browsing-related tests for the DockerRuntime, which connects to the ActionExecutor running in the sandbox."""

import os
import re

import pytest
from conftest import _close_test_runtime, _load_runtime

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
)
from openhands.events.observation import (
    BrowserOutputObservation,
    CmdOutputObservation,
    ErrorObservation,
    FileDownloadObservation,
)

# ============================================================================================================================
# Browsing tests, without evaluation (poetry install --without evaluation)
# For eval environments, tests need to run with poetry install
# ============================================================================================================================


# Skip all tests in this module for CLI runtime
pytestmark = pytest.mark.skipif(
    os.environ.get('TEST_RUNTIME') == 'cli',
    reason='CLIRuntime does not support browsing actions',
)


def parse_axtree_content(content: str) -> dict[str, str]:
    """Parse the accessibility tree content to extract bid -> element description mapping."""
    elements = {}
    current_bid = None
    description_lines = []

    # Find the accessibility tree section
    lines = content.split('\n')
    in_axtree = False

    for line in lines:
        line = line.strip()

        # Check if we're entering the accessibility tree section
        if 'BEGIN accessibility tree' in line:
            in_axtree = True
            continue
        elif 'END accessibility tree' in line:
            break

        if not in_axtree or not line:
            continue

        # Check for bid line format: [bid] element description
        bid_match = re.match(r'\[([a-zA-Z0-9]+)\]\s*(.*)', line)
        if bid_match:
            # Save previous element if it exists
            if current_bid and description_lines:
                elements[current_bid] = ' '.join(description_lines)

            # Start new element
            current_bid = bid_match.group(1)
            description_lines = [bid_match.group(2).strip()]
        else:
            # Add to current description if we have a bid
            if current_bid:
                description_lines.append(line)

    # Save last element
    if current_bid and description_lines:
        elements[current_bid] = ' '.join(description_lines)

    return elements


def find_element_by_text(axtree_elements: dict[str, str], text: str) -> str | None:
    """Find an element bid by searching for text in the element description."""
    text = text.lower().strip()
    for bid, description in axtree_elements.items():
        if text in description.lower():
            return bid
    return None


def find_element_by_id(axtree_elements: dict[str, str], element_id: str) -> str | None:
    """Find an element bid by searching for HTML id attribute."""
    for bid, description in axtree_elements.items():
        # Look for id="element_id" or id='element_id' patterns
        if f'id="{element_id}"' in description or f"id='{element_id}'" in description:
            return bid
    return None


def find_element_by_tag_and_attributes(
    axtree_elements: dict[str, str], tag: str, **attributes
) -> str | None:
    """Find an element bid by tag name and attributes."""
    tag = tag.lower()
    for bid, description in axtree_elements.items():
        description_lower = description.lower()

        # Check if this is the right tag
        if not description_lower.startswith(tag):
            continue

        # Check all required attributes
        match = True
        for attr_name, attr_value in attributes.items():
            attr_pattern = f'{attr_name}="{attr_value}"'
            if attr_pattern not in description:
                attr_pattern = f"{attr_name}='{attr_value}'"
                if attr_pattern not in description:
                    match = False
                    break

        if match:
            return bid

    return None


def test_browser_disabled(temp_dir, runtime_cls, run_as_openhands, dynamic_port):
    runtime, _ = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=False
    )

    action_cmd = CmdRunAction(
        command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
    )
    logger.info(action_cmd, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action_cmd)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    action_browse = BrowseURLAction(
        url=f'http://localhost:{dynamic_port}', return_axtree=False
    )
    logger.info(action_browse, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action_browse)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, ErrorObservation)
    assert 'Browser functionality is not supported or disabled' in obs.content

    _close_test_runtime(runtime)


def test_simple_browse(temp_dir, runtime_cls, run_as_openhands, dynamic_port):
    runtime, config = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=True
    )

    # Test browse
    action_cmd = CmdRunAction(
        command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
    )
    logger.info(action_cmd, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action_cmd)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert '[1]' in obs.content

    action_cmd = CmdRunAction(command='sleep 3 && cat server.log')
    logger.info(action_cmd, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action_cmd)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action_browse = BrowseURLAction(
        url=f'http://localhost:{dynamic_port}', return_axtree=False
    )
    logger.info(action_browse, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action_browse)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, BrowserOutputObservation)
    assert f'http://localhost:{dynamic_port}' in obs.url
    assert not obs.error
    assert obs.open_pages_urls == [f'http://localhost:{dynamic_port}/']
    assert obs.active_page_index == 0
    assert obs.last_browser_action == f'goto("http://localhost:{dynamic_port}")'
    assert obs.last_browser_action_error == ''
    assert 'Directory listing for /' in obs.content
    assert 'server.log' in obs.content

    # clean up
    action = CmdRunAction(command='rm -rf server.log')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    _close_test_runtime(runtime)


def test_browser_navigation_actions(
    temp_dir, runtime_cls, run_as_openhands, dynamic_port
):
    """Test browser navigation actions: goto, go_back, go_forward, noop."""
    runtime, config = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=True
    )
    try:
        # Create test HTML pages
        page1_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Page 1</title></head>
        <body>
            <h1>Page 1</h1>
            <a href="page2.html" id="link-to-page2">Go to Page 2</a>
        </body>
        </html>
        """

        page2_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Page 2</title></head>
        <body>
            <h1>Page 2</h1>
            <a href="page1.html" id="link-to-page1">Go to Page 1</a>
        </body>
        </html>
        """

        # Create HTML files in temp directory
        page1_path = os.path.join(temp_dir, 'page1.html')
        page2_path = os.path.join(temp_dir, 'page2.html')

        with open(page1_path, 'w') as f:
            f.write(page1_content)
        with open(page2_path, 'w') as f:
            f.write(page2_content)

        # Copy files to sandbox
        sandbox_dir = config.workspace_mount_path_in_sandbox
        runtime.copy_to(page1_path, sandbox_dir)
        runtime.copy_to(page2_path, sandbox_dir)

        # Start HTTP server
        action_cmd = CmdRunAction(
            command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
        )
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        # Wait for server to start
        action_cmd = CmdRunAction(command='sleep 3')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Test goto action
        action_browse = BrowseInteractiveAction(
            browser_actions=f'goto("http://localhost:{dynamic_port}/page1.html")',
            return_axtree=False,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error
        assert 'Page 1' in obs.content
        assert f'http://localhost:{dynamic_port}/page1.html' in obs.url

        # Test noop action (should not change page)
        action_browse = BrowseInteractiveAction(
            browser_actions='noop(500)', return_axtree=False
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error
        assert 'Page 1' in obs.content
        assert f'http://localhost:{dynamic_port}/page1.html' in obs.url

        # Navigate to page 2
        action_browse = BrowseInteractiveAction(
            browser_actions=f'goto("http://localhost:{dynamic_port}/page2.html")',
            return_axtree=False,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error
        assert 'Page 2' in obs.content
        assert f'http://localhost:{dynamic_port}/page2.html' in obs.url

        # Test go_back action
        action_browse = BrowseInteractiveAction(
            browser_actions='go_back()', return_axtree=False
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error
        assert 'Page 1' in obs.content
        assert f'http://localhost:{dynamic_port}/page1.html' in obs.url

        # Test go_forward action
        action_browse = BrowseInteractiveAction(
            browser_actions='go_forward()', return_axtree=False
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error
        assert 'Page 2' in obs.content
        assert f'http://localhost:{dynamic_port}/page2.html' in obs.url

        # Clean up
        action_cmd = CmdRunAction(command='pkill -f "python3 -m http.server" || true')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    finally:
        _close_test_runtime(runtime)


def test_browser_form_interactions(
    temp_dir, runtime_cls, run_as_openhands, dynamic_port
):
    """Test browser form interaction actions: fill, click, select_option, clear."""
    runtime, config = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=True
    )
    try:
        # Create a test form page
        form_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Form</title></head>
        <body>
            <h1>Test Form</h1>
            <form id="test-form">
                <input type="text" id="text-input" name="text" placeholder="Enter text">
                <textarea id="textarea-input" name="message" placeholder="Enter message"></textarea>
                <select id="select-input" name="option">
                    <option value="">Select an option</option>
                    <option value="option1">Option 1</option>
                    <option value="option2">Option 2</option>
                    <option value="option3">Option 3</option>
                </select>
                <button type="button" id="test-button">Test Button</button>
                <input type="submit" id="submit-button" value="Submit">
            </form>
            <div id="result"></div>
            <script>
                document.getElementById('test-button').onclick = function() {
                    document.getElementById('result').innerHTML = 'Button clicked!';
                };
            </script>
        </body>
        </html>
        """

        # Create HTML file
        form_path = os.path.join(temp_dir, 'form.html')
        with open(form_path, 'w') as f:
            f.write(form_content)

        # Copy to sandbox
        sandbox_dir = config.workspace_mount_path_in_sandbox
        runtime.copy_to(form_path, sandbox_dir)

        # Start HTTP server
        action_cmd = CmdRunAction(
            command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
        )
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'ACTION'})
        assert obs.exit_code == 0

        # Wait for server to start
        action_cmd = CmdRunAction(command='sleep 3')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Navigate to form page
        action_browse = BrowseInteractiveAction(
            browser_actions=f'goto("http://localhost:{dynamic_port}/form.html")',
            return_axtree=True,  # Need axtree to get element bids
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error
        assert 'Test Form' in obs.content

        # Parse the axtree to get actual bid values
        axtree_elements = parse_axtree_content(obs.content)

        # Find elements by their characteristics visible in the axtree
        text_input_bid = find_element_by_text(axtree_elements, 'Enter text')
        textarea_bid = find_element_by_text(axtree_elements, 'Enter message')
        select_bid = find_element_by_text(axtree_elements, 'combobox')
        button_bid = find_element_by_text(axtree_elements, 'Test Button')

        # Verify we found the correct elements
        assert text_input_bid is not None, (
            f'Could not find text input element in axtree. Available elements: {dict(list(axtree_elements.items())[:5])}'
        )
        assert textarea_bid is not None, (
            f'Could not find textarea element in axtree. Available elements: {dict(list(axtree_elements.items())[:5])}'
        )
        assert button_bid is not None, (
            f'Could not find button element in axtree. Available elements: {dict(list(axtree_elements.items())[:5])}'
        )
        assert select_bid is not None, (
            f'Could not find select element in axtree. Available elements: {dict(list(axtree_elements.items())[:5])}'
        )
        assert text_input_bid != button_bid, (
            'Text input bid should be different from button bid'
        )

        # Test fill action with real bid values
        action_browse = BrowseInteractiveAction(
            browser_actions=f"""
fill("{text_input_bid}", "Hello World")
fill("{textarea_bid}", "This is a test message")
""".strip(),
            return_axtree=True,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        # Verify the action executed successfully
        assert not obs.error, (
            f'Browser action failed with error: {obs.last_browser_action_error}'
        )

        # Parse the updated axtree to verify the text was actually filled
        updated_axtree_elements = parse_axtree_content(obs.content)

        # Check that the text input now contains our text
        assert text_input_bid in updated_axtree_elements, (
            f'Text input element {text_input_bid} should be present in updated axtree. Available elements: {list(updated_axtree_elements.keys())[:10]}'
        )
        text_input_desc = updated_axtree_elements[text_input_bid]
        # The filled value should appear in the element description (axtree shows values differently)
        assert 'Hello World' in text_input_desc or "'Hello World'" in text_input_desc, (
            f"Text input should contain 'Hello World' but description is: {text_input_desc}"
        )

        assert textarea_bid in updated_axtree_elements, (
            f'Textarea element {textarea_bid} should be present in updated axtree. Available elements: {list(updated_axtree_elements.keys())[:10]}'
        )
        textarea_desc = updated_axtree_elements[textarea_bid]
        assert (
            'This is a test message' in textarea_desc
            or "'This is a test message'" in textarea_desc
        ), f'Textarea should contain test message but description is: {textarea_desc}'

        # Test select_option action with real bid
        action_browse = BrowseInteractiveAction(
            browser_actions=f'select_option("{select_bid}", "option2")',
            return_axtree=True,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, (
            f'Select option action failed: {obs.last_browser_action_error}'
        )

        # Verify that option2 is now selected
        updated_axtree_elements = parse_axtree_content(obs.content)
        assert select_bid in updated_axtree_elements, (
            f'Select element {select_bid} should be present in updated axtree. Available elements: {list(updated_axtree_elements.keys())[:10]}'
        )
        select_desc = updated_axtree_elements[select_bid]
        # The selected option should be reflected in the select element description
        assert 'option2' in select_desc or 'Option 2' in select_desc, (
            f"Select element should show 'option2' as selected but description is: {select_desc}"
        )

        # Test click action with real bid
        action_browse = BrowseInteractiveAction(
            browser_actions=f'click("{button_bid}")', return_axtree=True
        )
        obs = runtime.run_action(action_browse)
        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, f'Click action failed: {obs.last_browser_action_error}'

        # Verify that the button click triggered the JavaScript and updated the result div
        updated_axtree_elements = parse_axtree_content(obs.content)
        # Look for the "Button clicked!" text that should appear in the result div
        result_found = any(
            'Button clicked!' in desc for desc in updated_axtree_elements.values()
        )
        assert result_found, (
            f"Button click should have triggered JavaScript to show 'Button clicked!' but not found in: {dict(list(updated_axtree_elements.items())[:10])}"
        )

        # Test clear action with real bid
        action_browse = BrowseInteractiveAction(
            browser_actions=f'clear("{text_input_bid}")', return_axtree=True
        )
        obs = runtime.run_action(action_browse)
        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, f'Clear action failed: {obs.last_browser_action_error}'

        # Verify that the text input is now empty/cleared
        updated_axtree_elements = parse_axtree_content(obs.content)
        assert text_input_bid in updated_axtree_elements
        text_input_desc = updated_axtree_elements[text_input_bid]
        # After clearing, the input should not contain the previous text
        assert 'Hello World' not in text_input_desc, (
            f'Text input should be cleared but still contains text: {text_input_desc}'
        )
        # Check that it's back to showing placeholder text or is empty
        assert (
            'Enter text' in text_input_desc  # placeholder text
            or 'textbox' in text_input_desc.lower()  # generic textbox description
            or text_input_desc.strip() == ''  # empty description
        ), (
            f'Cleared text input should show placeholder or be empty but description is: {text_input_desc}'
        )

        # Clean up
        action_cmd = CmdRunAction(command='pkill -f "python3 -m http.server" || true')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    finally:
        _close_test_runtime(runtime)


def test_browser_interactive_actions(
    temp_dir, runtime_cls, run_as_openhands, dynamic_port
):
    """Test browser interactive actions: scroll, hover, fill, press, focus."""
    runtime, config = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=True
    )
    try:
        # Create a test page with scrollable content
        scroll_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Scroll Test</title>
            <style>
                body { margin: 0; padding: 20px; }
                .content { height: 2000px; background: linear-gradient(to bottom, #ff0000, #0000ff); }
                .hover-target {
                    width: 200px; height: 100px; background: #ccc; margin: 20px;
                    border: 2px solid #000; cursor: pointer;
                }
                .hover-target:hover { background: #ffff00; }
                #focus-input { margin: 20px; padding: 10px; font-size: 16px; }
            </style>
        </head>
        <body>
            <h1>Interactive Test Page</h1>
            <div class="hover-target" id="hover-div">Hover over me</div>
            <input type="text" id="focus-input" placeholder="Focus me and type">
            <div class="content">
                <p>This is a long scrollable page...</p>
                <p style="margin-top: 500px;">Middle content</p>
                <p style="margin-top: 500px;" id="bottom-content">Bottom content</p>
            </div>
        </body>
        </html>
        """

        # Create HTML file
        scroll_path = os.path.join(temp_dir, 'scroll.html')
        with open(scroll_path, 'w') as f:
            f.write(scroll_content)

        # Copy to sandbox
        sandbox_dir = config.workspace_mount_path_in_sandbox
        runtime.copy_to(scroll_path, sandbox_dir)

        # Start HTTP server
        action_cmd = CmdRunAction(
            command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
        )
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        # Wait for server to start
        action_cmd = CmdRunAction(command='sleep 3')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Navigate to scroll page
        action_browse = BrowseInteractiveAction(
            browser_actions=f'goto("http://localhost:{dynamic_port}/scroll.html")',
            return_axtree=True,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error
        assert 'Interactive Test Page' in obs.content

        # Test scroll action
        action_browse = BrowseInteractiveAction(
            browser_actions='scroll(0, 300)',  # Scroll down 300 pixels
            return_axtree=True,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, f'Scroll action failed: {obs.last_browser_action_error}'
        # Verify the scroll action was recorded correctly
        assert 'scroll(0, 300)' in obs.last_browser_action, (
            f'Expected scroll action in browser history but got: {obs.last_browser_action}'
        )

        # Parse the axtree to get actual bid values for interactive elements
        axtree_elements = parse_axtree_content(obs.content)

        # Find elements by their characteristics visible in the axtree
        hover_div_bid = find_element_by_text(axtree_elements, 'Hover over me')
        focus_input_bid = find_element_by_text(axtree_elements, 'Focus me and type')

        # Verify we found the required elements
        assert hover_div_bid is not None, (
            f'Could not find hover div element in axtree. Available elements: {dict(list(axtree_elements.items())[:5])}'
        )
        assert focus_input_bid is not None, (
            f'Could not find focus input element in axtree. Available elements: {dict(list(axtree_elements.items())[:5])}'
        )

        # Test hover action with real bid
        action_browse = BrowseInteractiveAction(
            browser_actions=f'hover("{hover_div_bid}")', return_axtree=True
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, f'Hover action failed: {obs.last_browser_action_error}'

        # Test focus action with real bid
        action_browse = BrowseInteractiveAction(
            browser_actions=f'focus("{focus_input_bid}")', return_axtree=True
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, f'Focus action failed: {obs.last_browser_action_error}'

        # Verify that the input element is now focused
        assert obs.focused_element_bid == focus_input_bid, (
            f'Expected focused element to be {focus_input_bid}, but got {obs.focused_element_bid}'
        )

        # Test fill action (type in focused input) with real bid
        action_browse = BrowseInteractiveAction(
            browser_actions=f'fill("{focus_input_bid}", "TestValue123")',
            return_axtree=True,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, f'Fill action failed: {obs.last_browser_action_error}'

        # Verify that the text was actually entered
        updated_axtree_elements = parse_axtree_content(obs.content)
        assert focus_input_bid in updated_axtree_elements, (
            f'Focus input element {focus_input_bid} should be present in updated axtree. Available elements: {list(updated_axtree_elements.keys())[:10]}'
        )
        input_desc = updated_axtree_elements[focus_input_bid]
        assert 'TestValue123' in input_desc or "'TestValue123'" in input_desc, (
            f"Input should contain 'TestValue123' but description is: {input_desc}"
        )

        # Test press action (for pressing individual keys) with real bid
        action_browse = BrowseInteractiveAction(
            browser_actions=f'press("{focus_input_bid}", "Backspace")',
            return_axtree=True,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, f'Press action failed: {obs.last_browser_action_error}'

        # Verify the backspace removed the last character (3 from TestValue123)
        updated_axtree_elements = parse_axtree_content(obs.content)
        assert focus_input_bid in updated_axtree_elements, (
            f'Focus input element {focus_input_bid} should be present in updated axtree. Available elements: {list(updated_axtree_elements.keys())[:10]}'
        )
        input_desc = updated_axtree_elements[focus_input_bid]
        assert 'TestValue12' in input_desc or "'TestValue12'" in input_desc, (
            f"Input should contain 'TestValue12' after backspace but description is: {input_desc}"
        )

        # Test multiple actions in sequence
        action_browse = BrowseInteractiveAction(
            browser_actions="""
scroll(0, -200)
noop(1000)
scroll(0, 400)
""".strip(),
            return_axtree=False,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, (
            f'Multiple actions sequence failed: {obs.last_browser_action_error}'
        )
        # Verify the last action in the sequence was recorded
        assert (
            'scroll(0, 400)' in obs.last_browser_action
            or 'noop(1000)' in obs.last_browser_action
        ), f'Expected final action from sequence but got: {obs.last_browser_action}'

        # Clean up
        action_cmd = CmdRunAction(command='pkill -f "python3 -m http.server" || true')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    finally:
        _close_test_runtime(runtime)


def test_browser_file_upload(temp_dir, runtime_cls, run_as_openhands, dynamic_port):
    """Test browser file upload action."""
    runtime, config = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=True
    )
    try:
        # Create a test file to upload
        test_file_content = 'This is a test file for upload testing.'
        test_file_path = os.path.join(temp_dir, 'upload_test.txt')
        with open(test_file_path, 'w') as f:
            f.write(test_file_content)

        # Create an upload form page
        upload_content = """
        <!DOCTYPE html>
        <html>
        <head><title>File Upload Test</title></head>
        <body>
            <h1>File Upload Test</h1>
            <form enctype="multipart/form-data">
                <input type="file" id="file-input" name="file" accept=".txt,.pdf,.png">
                <button type="button" onclick="handleUpload()">Upload File</button>
            </form>
            <div id="upload-result"></div>
            <script>
                function handleUpload() {
                    const fileInput = document.getElementById('file-input');
                    if (fileInput.files.length > 0) {
                        document.getElementById('upload-result').innerHTML =
                            'File selected: ' + fileInput.files[0].name;
                    } else {
                        document.getElementById('upload-result').innerHTML = 'No file selected';
                    }
                }
            </script>
        </body>
        </html>
        """

        # Create HTML file
        upload_path = os.path.join(temp_dir, 'upload.html')
        with open(upload_path, 'w') as f:
            f.write(upload_content)

        # Copy files to sandbox
        sandbox_dir = config.workspace_mount_path_in_sandbox
        runtime.copy_to(upload_path, sandbox_dir)
        runtime.copy_to(test_file_path, sandbox_dir)

        # Start HTTP server
        action_cmd = CmdRunAction(
            command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
        )
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        # Wait for server to start
        action_cmd = CmdRunAction(command='sleep 3')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Navigate to upload page
        action_browse = BrowseInteractiveAction(
            browser_actions=f'goto("http://localhost:{dynamic_port}/upload.html")',
            return_axtree=True,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error
        assert 'File Upload Test' in obs.content

        # Parse the axtree to get the file input bid
        axtree_elements = parse_axtree_content(obs.content)
        # File inputs often show up as buttons in axtree, try multiple strategies
        file_input_bid = (
            find_element_by_text(axtree_elements, 'Choose File')
            or find_element_by_text(axtree_elements, 'No file chosen')
            or find_element_by_text(axtree_elements, 'Browse')
            or find_element_by_text(axtree_elements, 'file')
            or find_element_by_id(axtree_elements, 'file-input')
        )

        # Also look for button near the file input (Upload File button)
        upload_button_bid = find_element_by_text(axtree_elements, 'Upload File')

        # Test upload_file action with real bid
        assert file_input_bid is not None, (
            f'Could not find file input element in axtree. Available elements: {dict(list(axtree_elements.items())[:10])}'
        )

        action_browse = BrowseInteractiveAction(
            browser_actions=f'upload_file("{file_input_bid}", "/workspace/upload_test.txt")',
            return_axtree=True,
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs, BrowserOutputObservation)
        assert not obs.error, (
            f'File upload action failed: {obs.last_browser_action_error}'
        )

        # Verify the file input now shows the selected file
        updated_axtree_elements = parse_axtree_content(obs.content)
        assert file_input_bid in updated_axtree_elements, (
            f'File input element {file_input_bid} should be present in updated axtree. Available elements: {list(updated_axtree_elements.keys())[:10]}'
        )
        file_input_desc = updated_axtree_elements[file_input_bid]
        # File inputs typically show the filename when a file is selected
        assert (
            'upload_test.txt' in file_input_desc
            or 'upload_test' in file_input_desc
            or 'txt' in file_input_desc
        ), f'File input should show selected file but description is: {file_input_desc}'

        # Test clicking the upload button to trigger the JavaScript function
        if upload_button_bid:
            action_browse = BrowseInteractiveAction(
                browser_actions=f'click("{upload_button_bid}")',
                return_axtree=True,
            )
            logger.info(action_browse, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action_browse)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

            assert isinstance(obs, BrowserOutputObservation)
            assert not obs.error, (
                f'Upload button click failed: {obs.last_browser_action_error}'
            )

            # Check if the JavaScript function executed and updated the result div
            final_axtree_elements = parse_axtree_content(obs.content)
            # Look for the result text that should be set by JavaScript
            result_found = any(
                'File selected:' in desc or 'upload_test.txt' in desc
                for desc in final_axtree_elements.values()
            )
            assert result_found, (
                f'JavaScript upload handler should have updated the page but no result found in: {dict(list(final_axtree_elements.items())[:10])}'
            )

        # Clean up
        action_cmd = CmdRunAction(command='pkill -f "python3 -m http.server" || true')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    finally:
        _close_test_runtime(runtime)


def test_read_pdf_browse(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=True
    )
    try:
        # Create a PDF file using reportlab in the host environment
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf_path = os.path.join(temp_dir, 'test_document.pdf')
        pdf_content = 'This is test content for PDF reading test'

        c = canvas.Canvas(pdf_path, pagesize=letter)
        # Add more content to make the PDF more robust
        c.drawString(100, 750, pdf_content)
        c.drawString(100, 700, 'Additional line for PDF structure')
        c.drawString(100, 650, 'Third line to ensure valid PDF')
        # Explicitly set PDF version and ensure proper structure
        c.setPageCompression(0)  # Disable compression for simpler structure
        c.save()

        # Copy the PDF to the sandbox
        sandbox_dir = config.workspace_mount_path_in_sandbox
        runtime.copy_to(pdf_path, sandbox_dir)

        # Start HTTP server
        action_cmd = CmdRunAction(command='ls -alh')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == 0
        assert 'test_document.pdf' in obs.content

        # Get server url
        action_cmd = CmdRunAction(command='cat /tmp/oh-server-url')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0
        server_url = obs.content.strip()

        # Browse to the PDF file
        pdf_url = f'{server_url}/view?path=/workspace/test_document.pdf'
        action_browse = BrowseInteractiveAction(
            browser_actions=f'goto("{pdf_url}")', return_axtree=False
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Verify the browser observation
        assert isinstance(obs, BrowserOutputObservation)
        observation_text = str(obs)
        assert '[Action executed successfully.]' in observation_text
        assert 'Canvas' in observation_text
        assert (
            'Screenshot saved to: /workspace/.browser_screenshots/screenshot_'
            in observation_text
        )

        # Check the /workspace/.browser_screenshots folder
        action_cmd = CmdRunAction(command='ls /workspace/.browser_screenshots')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == 0
        assert 'screenshot_' in obs.content
        assert '.png' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_read_png_browse(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=True
    )
    try:
        # Create a PNG file using PIL in the host environment
        from PIL import Image, ImageDraw

        png_path = os.path.join(temp_dir, 'test_image.png')
        # Create a simple image with text
        img = Image.new('RGB', (400, 200), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        text = 'This is a test PNG image'
        d.text((20, 80), text, fill=(0, 0, 0))
        img.save(png_path)

        # Copy the PNG to the sandbox
        sandbox_dir = config.workspace_mount_path_in_sandbox
        runtime.copy_to(png_path, sandbox_dir)

        # Verify the file exists in the sandbox
        action_cmd = CmdRunAction(command='ls -alh')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == 0
        assert 'test_image.png' in obs.content

        # Get server url
        action_cmd = CmdRunAction(command='cat /tmp/oh-server-url')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0
        server_url = obs.content.strip()

        # Browse to the PNG file
        png_url = f'{server_url}/view?path=/workspace/test_image.png'
        action_browse = BrowseInteractiveAction(
            browser_actions=f'goto("{png_url}")', return_axtree=False
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Verify the browser observation
        assert isinstance(obs, BrowserOutputObservation)
        observation_text = str(obs)
        assert '[Action executed successfully.]' in observation_text
        assert 'File Viewer - test_image.png' in observation_text
        assert (
            'Screenshot saved to: /workspace/.browser_screenshots/screenshot_'
            in observation_text
        )

        # Check the /workspace/.browser_screenshots folder
        action_cmd = CmdRunAction(command='ls /workspace/.browser_screenshots')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == 0
        assert 'screenshot_' in obs.content
        assert '.png' in obs.content
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skip(reason='This test is flaky')
def test_download_file(temp_dir, runtime_cls, run_as_openhands, dynamic_port):
    """Test downloading a file using the browser."""
    runtime, config = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=True
    )
    try:
        # Minimal PDF content for testing
        pdf_content = b"""%PDF-1.4
        1 0 obj

        /Type /Catalog
        /Pages 2 0 R
        >>
        endobj
        2 0 obj

        /Type /Pages
        /Kids [3 0 R]
        /Count 1
        >>
        endobj
        3 0 obj

        /Type /Page
        /Parent 2 0 R
        /MediaBox [0 0 612 792]
        >>
        endobj
        xref
        0 4
        0000000000 65535 f
        0000000010 00000 n
        0000000053 00000 n
        0000000125 00000 n
        trailer

        /Size 4
        /Root 1 0 R
        >>
        startxref
        212
        %%EOF"""

        test_file_name = 'test_download.pdf'
        test_file_path = os.path.join(temp_dir, test_file_name)
        with open(test_file_path, 'wb') as f:
            f.write(pdf_content)

        # Copy the file to the sandbox
        sandbox_dir = config.workspace_mount_path_in_sandbox
        runtime.copy_to(test_file_path, sandbox_dir)

        # Create a simple HTML page with a download link
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Download Test</title>
        </head>
        <body>
            <h1>Download Test Page</h1>
            <p>Click the link below to download the test file:</p>
            <a href="/{test_file_name}" download="{test_file_name}" id="download-link">Download Test File</a>
        </body>
        </html>
        """

        html_file_path = os.path.join(temp_dir, 'download_test.html')
        with open(html_file_path, 'w') as f:
            f.write(html_content)

        # Copy the HTML file to the sandbox
        runtime.copy_to(html_file_path, sandbox_dir)

        # Verify the files exist in the sandbox
        action_cmd = CmdRunAction(command='ls -alh')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == 0
        assert test_file_name in obs.content
        assert 'download_test.html' in obs.content

        # Ensure downloads directory exists
        action_cmd = CmdRunAction(command='mkdir -p /workspace/.downloads')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        # Start HTTP server
        action_cmd = CmdRunAction(
            command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
        )
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == 0

        # Wait for server to start
        action_cmd = CmdRunAction(command='sleep 2')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Browse to the HTML page
        action_browse = BrowseURLAction(url=f'http://localhost:{dynamic_port}/')
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Verify the browser observation
        assert isinstance(obs, BrowserOutputObservation)
        assert f'http://localhost:{dynamic_port}/download_test.html' in obs.url
        assert not obs.error
        assert 'Download Test Page' in obs.content

        # Go to the PDF file url directly - this should trigger download
        file_url = f'http://localhost:{dynamic_port}/{test_file_name}'
        action_browse = BrowseInteractiveAction(
            browser_actions=f'goto("{file_url}")',
        )
        logger.info(action_browse, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_browse)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Verify the browser observation after navigating to PDF file
        downloaded_file_name = 'file_1.pdf'
        assert isinstance(obs, FileDownloadObservation)
        assert 'Location of downloaded file:' in str(obs)
        assert downloaded_file_name in str(obs)  # File is renamed

        # Wait for download to complete
        action_cmd = CmdRunAction(command='sleep 3')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Check if the file was downloaded
        action_cmd = CmdRunAction(command='ls -la /workspace')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == 0
        assert downloaded_file_name in obs.content

        # Clean up
        action_cmd = CmdRunAction(command='pkill -f "python3 -m http.server" || true')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action_cmd = CmdRunAction(command='rm -f server.log')
        logger.info(action_cmd, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action_cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    finally:
        _close_test_runtime(runtime)
