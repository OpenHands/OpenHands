# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import base64
import datetime
import os
from pathlib import Path
from typing import Any

from browsergym.utils.obs import flatten_axtree_to_str
from PIL import Image

from openhands.core.exceptions import BrowserUnavailableException
from openhands.core.schema import ActionType
from openhands.events.action import BrowseInteractiveAction, BrowseURLAction
from openhands.events.observation import BrowserOutputObservation
from openhands.runtime.browser.base64 import png_base64_url_to_image
from openhands.runtime.browser.browser_env import BrowserEnv
from openhands.utils.async_utils import call_sync_from_async


def get_axtree_str(
    axtree_object: dict[str, Any],
    extra_element_properties: dict[str, Any],
    filter_visible_only: bool = False,
) -> str:
    cur_axtree_txt = flatten_axtree_to_str(
        axtree_object,
        extra_properties=extra_element_properties,
        with_clickable=True,
        skip_generic=False,
        filter_visible_only=filter_visible_only,
    )
    return str(cur_axtree_txt)


def get_agent_obs_text(obs: BrowserOutputObservation) -> str:
    """Get a concise text that will be shown to the agent."""
    if obs.trigger_by_action == ActionType.BROWSE_INTERACTIVE:
        text = f'[Current URL: {obs.url}]\n'
        text += f'[Focused element bid: {obs.focused_element_bid}]\n'

        # Add screenshot path information if available
        if obs.screenshot_path:
            text += f'[Screenshot saved to: {obs.screenshot_path}]\n'

        text += '\n'

        if obs.error:
            text += (
                '================ BEGIN error message ===============\n'
                'The following error occurred when executing the last action:\n'
                f'{obs.last_browser_action_error}\n'
                '================ END error message ===============\n'
            )
        else:
            text += '[Action executed successfully.]\n'
        try:
            # We do not filter visible only here because we want to show the full content
            # of the web page to the agent for simplicity.
            # FIXME: handle the case when the web page is too large
            cur_axtree_txt = get_axtree_str(
                obs.axtree_object,
                obs.extra_element_properties,
                filter_visible_only=obs.filter_visible_only,
            )
            if not obs.filter_visible_only:
                text += (
                    f'Accessibility tree of the COMPLETE webpage:\nNote: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.\n'
                    f'============== BEGIN accessibility tree ==============\n'
                    f'{cur_axtree_txt}\n'
                    f'============== END accessibility tree ==============\n'
                )
            else:
                text += (
                    f'Accessibility tree of the VISIBLE portion of the webpage (accessibility tree of complete webpage is too large and you may need to scroll to view remaining portion of the webpage):\nNote: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.\n'
                    f'============== BEGIN accessibility tree ==============\n'
                    f'{cur_axtree_txt}\n'
                    f'============== END accessibility tree ==============\n'
                )
        except Exception as e:
            text += f'\n[Error encountered when processing the accessibility tree: {e}]'
        return text

    elif obs.trigger_by_action == ActionType.BROWSE:
        text = f'[Current URL: {obs.url}]\n'

        if obs.error:
            text += (
                '================ BEGIN error message ===============\n'
                'The following error occurred when trying to visit the URL:\n'
                f'{obs.last_browser_action_error}\n'
                '================ END error message ===============\n'
            )
        text += '============== BEGIN webpage content ==============\n'
        text += obs.content
        text += '\n============== END webpage content ==============\n'
        return text
    else:
        raise ValueError(f'Invalid trigger_by_action: {obs.trigger_by_action}')


async def browse(
    action: BrowseURLAction | BrowseInteractiveAction,
    browser: BrowserEnv | None,
    workspace_dir: str | None = None,
) -> BrowserOutputObservation:
    if browser is None:
        raise BrowserUnavailableException()

    if isinstance(action, BrowseURLAction):
        # legacy BrowseURLAction
        asked_url = action.url
        if not asked_url.startswith('http'):
            asked_url = os.path.abspath(os.curdir) + action.url
        action_str = f'goto("{asked_url}")'

    elif isinstance(action, BrowseInteractiveAction):
        # new BrowseInteractiveAction, supports full featured BrowserGym actions
        # action in BrowserGym: see https://github.com/ServiceNow/BrowserGym/blob/main/core/src/browsergym/core/action/functions.py
        action_str = action.browser_actions
    else:
        raise ValueError(f'Invalid action type: {action.action}')

    try:
        # obs provided by BrowserGym: see https://github.com/ServiceNow/BrowserGym/blob/main/core/src/browsergym/core/env.py#L396
        obs = await call_sync_from_async(browser.step, action_str)

        # Save screenshot if workspace_dir is provided
        screenshot_path = None
        if workspace_dir is not None and obs.get('screenshot'):
            # Create screenshots directory if it doesn't exist
            screenshots_dir = Path(workspace_dir) / '.browser_screenshots'
            screenshots_dir.mkdir(exist_ok=True)

            # Generate a filename based on timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            screenshot_filename = f'screenshot_{timestamp}.png'
            screenshot_path = str(screenshots_dir / screenshot_filename)

            # Direct image saving from base64 data without using PIL's Image.open
            # This approach bypasses potential encoding issues that might occur when
            # converting between different image representations, ensuring the raw PNG
            # data from the browser is saved directly to disk.

            # Extract the base64 data
            base64_data = obs.get('screenshot', '')
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]

            try:
                # Decode base64 directly to binary
                image_data = base64.b64decode(base64_data)

                # Write binary data directly to file
                with open(screenshot_path, 'wb') as f:
                    f.write(image_data)

                # Verify the image was saved correctly by opening it
                # This is just a verification step and can be removed in production
                Image.open(screenshot_path).verify()
            except Exception:
                # If direct saving fails, fall back to the original method
                image = png_base64_url_to_image(obs.get('screenshot'))
                image.save(screenshot_path, format='PNG', optimize=True)

        # Create the observation with all data
        observation = BrowserOutputObservation(
            content=obs['text_content'],  # text content of the page
            url=obs.get('url', ''),  # URL of the page
            screenshot=obs.get('screenshot', None),  # base64-encoded screenshot, png
            screenshot_path=screenshot_path,  # path to saved screenshot file
            set_of_marks=obs.get(
                'set_of_marks', None
            ),  # base64-encoded Set-of-Marks annotated screenshot, png,
            goal_image_urls=obs.get('image_content', []),
            open_pages_urls=obs.get('open_pages_urls', []),  # list of open pages
            active_page_index=obs.get(
                'active_page_index', -1
            ),  # index of the active page
            axtree_object=obs.get('axtree_object', {}),  # accessibility tree object
            extra_element_properties=obs.get('extra_element_properties', {}),
            focused_element_bid=obs.get(
                'focused_element_bid', None
            ),  # focused element bid
            last_browser_action=obs.get(
                'last_action', ''
            ),  # last browser env action performed
            last_browser_action_error=obs.get('last_action_error', ''),
            error=True if obs.get('last_action_error', '') else False,  # error flag
            trigger_by_action=action.action,
        )

        # Process the content first using the axtree_object
        observation.content = get_agent_obs_text(observation)

        # If return_axtree is False, remove the axtree_object to save space
        if not action.return_axtree:
            observation.dom_object = {}
            observation.axtree_object = {}
            observation.extra_element_properties = {}

        return observation
    except Exception as e:
        error_message = str(e)
        error_url = asked_url if action.action == ActionType.BROWSE else ''

        # Create error observation
        observation = BrowserOutputObservation(
            content=error_message,
            screenshot='',
            screenshot_path=None,
            error=True,
            last_browser_action_error=error_message,
            url=error_url,
            trigger_by_action=action.action,
        )

        # Process the content using get_agent_obs_text regardless of return_axtree value
        try:
            observation.content = get_agent_obs_text(observation)
        except Exception:
            # If get_agent_obs_text fails, keep the original error message
            pass

        return observation
