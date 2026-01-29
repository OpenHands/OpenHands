# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str

from openhands.agenthub.browsing_agent.response_parser import BrowsingResponseParser
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    Action,
    AgentFinishAction,
    BrowseInteractiveAction,
    MessageAction,
)
from openhands.events.event import EventSource
from openhands.events.observation import BrowserOutputObservation
from openhands.events.observation.observation import Observation
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.plugins import (
    PluginRequirement,
)


def get_error_prefix(obs: BrowserOutputObservation) -> str:
    # temporary fix for OneStopMarket to ignore timeout errors
    if 'timeout' in obs.last_browser_action_error:
        return ''
    return f'## Error from previous action:\n{obs.last_browser_action_error}\n'


def create_goal_prompt(
    goal: str, image_urls: list[str] | None
) -> tuple[str, list[str]]:
    goal_txt: str = f"""\
# Instructions
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}
"""
    goal_image_urls = []
    if image_urls is not None:
        for idx, url in enumerate(image_urls):
            goal_txt = goal_txt + f'Images: Goal input image ({idx + 1})\n'
            goal_image_urls.append(url)
    goal_txt += '\n'
    return goal_txt, goal_image_urls


def create_observation_prompt(
    axtree_txt: str,
    tabs: str,
    focused_element: str,
    error_prefix: str,
    som_screenshot: str | None,
) -> tuple[str, str | None]:
    txt_observation = f"""
# Observation of current step:
{tabs}{axtree_txt}{focused_element}{error_prefix}
"""

    # screenshot + som: will be a non-empty string if present in observation
    screenshot_url = None
    if (som_screenshot is not None) and (len(som_screenshot) > 0):
        txt_observation += 'Image: Current page screenshot (Note that only visible portion of webpage is present in the screenshot. You may need to scroll to view the remaining portion of the web-page.\n'
        screenshot_url = som_screenshot
    else:
        logger.info('SOM Screenshot not present in observation!')
    txt_observation += '\n'
    return txt_observation, screenshot_url


def get_tabs(obs: BrowserOutputObservation) -> str:
    prompt_pieces = ['\n## Currently open tabs:']
    for page_index, page_url in enumerate(obs.open_pages_urls):
        active_or_not = ' (active tab)' if page_index == obs.active_page_index else ''
        prompt_piece = f"""\
Tab {page_index}{active_or_not}:
URL: {page_url}
"""
        prompt_pieces.append(prompt_piece)
    return '\n'.join(prompt_pieces) + '\n'


def get_axtree(axtree_txt: str) -> str:
    bid_info = """\
Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.

"""
    visible_tag_info = """\
Note: You can only interact with visible elements. If the "visible" tag is not present, the element is not visible on the page.

"""
    return f'\n## AXTree:\n{bid_info}{visible_tag_info}{axtree_txt}\n'


def get_action_prompt(action_set: HighLevelActionSet) -> str:
    action_set_generic_info = """\
Note: This action set allows you to interact with your environment. Most of them are python function executing playwright code. The primary way of referring to elements in the page is through bid which are specified in your observations.

"""
    action_description = action_set.describe(
        with_long_description=False,
        with_examples=False,
    )
    action_prompt = f'# Action space:\n{action_set_generic_info}{action_description}\n'
    return action_prompt


def get_history_prompt(prev_actions: list[BrowseInteractiveAction]) -> str:
    history_prompt = ['# History of all previous interactions with the task:\n']
    for i in range(len(prev_actions)):
        history_prompt.append(f'## step {i + 1}')
        history_prompt.append(
            f'\nOuput thought and action: {prev_actions[i].thought} ```{prev_actions[i].browser_actions}```\n'
        )
    return '\n'.join(history_prompt) + '\n'


class VisualBrowsingAgent(Agent):
    VERSION = '1.0'
    """
    VisualBrowsing Agent that can uses webpage screenshots during browsing.
    """

    sandbox_plugins: list[PluginRequirement] = []
    response_parser = BrowsingResponseParser()

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the VisualBrowsingAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(config, llm_registry)
        # define a configurable action space, with chat functionality, web navigation, and webpage grounding using accessibility tree and HTML.
        # see https://github.com/ServiceNow/BrowserGym/blob/main/core/src/browsergym/core/action/highlevel.py for more details
        action_subsets = [
            'chat',
            'bid',
            'nav',
            'tab',
            'infeas',
        ]
        self.action_space = HighLevelActionSet(
            subsets=action_subsets,
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,
        )
        self.action_prompt = get_action_prompt(self.action_space)
        self.abstract_example = f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:

You must mandatorily think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect that your previous action had on the current content of the page. In summary the next action I will perform is ```{self.action_space.example_action(abstract=True)}```
"""
        self.concrete_example = """
# Concrete Example

Here is a concrete example of how to format your answer. Make sure to generate the action in the correct format ensuring that the action is present inside ``````:

Let's think step-by-step. From previous action I tried to set the value of year to "2022", using select_option, but it doesn't appear to be in the form. It may be a dynamic dropdown, I will try using click with the bid "324" and look at the response from the page. In summary the next action I will perform is ```click('324')```
"""
        self.hints = """
Note:
* Make sure to use bid to identify elements when using commands.
* Interacting with combobox, dropdowns and auto-complete fields can be tricky, sometimes you need to use select_option, while other times you need to use fill or click and wait for the reaction of the page.

"""
        self.reset()

    def reset(self) -> None:
        """Resets the VisualBrowsingAgent's internal state."""
        super().reset()
        # Reset agent-specific counters but not LLM metrics
        self.error_accumulator = 0

    def step(self, state: State) -> Action:
        """Performs one step using the VisualBrowsingAgent.

        This includes gathering information on previous steps and prompting the model to make a browsing command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - BrowseInteractiveAction(browsergym_command) - BrowserGym commands to run
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        messages: list[Message] = []
        prev_actions = []
        cur_axtree_txt = ''
        error_prefix = ''
        focused_element = ''
        tabs = ''
        last_obs = None
        last_action = None
        set_of_marks = None  # Initialize set_of_marks to None

        if len(state.view) == 1:
            # for visualwebarena, webarena and miniwob++ eval, we need to retrieve the initial observation already in browser env
            # initialize and retrieve the first observation by issuing an noop OP
            # For non-benchmark browsing, the browser env starts with a blank page, and the agent is expected to first navigate to desired websites
            return BrowseInteractiveAction(
                browser_actions='noop(1000)', return_axtree=True
            )

        for event in state.view:
            if isinstance(event, BrowseInteractiveAction):
                prev_actions.append(event)
                last_action = event
            elif isinstance(event, MessageAction) and event.source == EventSource.AGENT:
                # agent has responded, task finished.
                return AgentFinishAction(outputs={'content': event.content})
            elif isinstance(event, Observation):
                # Only process BrowserOutputObservation and skip other observation types
                if not isinstance(event, BrowserOutputObservation):
                    continue
                last_obs = event

        if len(prev_actions) >= 1:  # ignore noop()
            prev_actions = prev_actions[1:]  # remove the first noop action

        # if the final BrowserInteractiveAction exec BrowserGym's send_msg_to_user,
        # we should also send a message back to the user in OpenHands and call it a day
        if (
            isinstance(last_action, BrowseInteractiveAction)
            and last_action.browsergym_send_msg_to_user
        ):
            return MessageAction(last_action.browsergym_send_msg_to_user)

        history_prompt = get_history_prompt(prev_actions)
        if isinstance(last_obs, BrowserOutputObservation):
            if last_obs.error:
                # add error recovery prompt prefix
                error_prefix = get_error_prefix(last_obs)
                if len(error_prefix) > 0:
                    self.error_accumulator += 1
                    if self.error_accumulator > 5:
                        return MessageAction(
                            'Too many errors encountered. Task failed.'
                        )
            focused_element = '## Focused element:\nNone\n'
            if last_obs.focused_element_bid is not None:
                focused_element = (
                    f"## Focused element:\nbid='{last_obs.focused_element_bid}'\n"
                )
            tabs = get_tabs(last_obs)
            try:
                # IMPORTANT: keep AX Tree of full webpage, add visible and clickable tags
                cur_axtree_txt = flatten_axtree_to_str(
                    last_obs.axtree_object,
                    extra_properties=last_obs.extra_element_properties,
                    with_visible=True,
                    with_clickable=True,
                    with_center_coords=False,
                    with_bounding_box_coords=False,
                    filter_visible_only=False,
                    filter_with_bid_only=False,
                    filter_som_only=False,
                )
                cur_axtree_txt = get_axtree(axtree_txt=cur_axtree_txt)
            except Exception as e:
                logger.error(
                    'Error when trying to process the accessibility tree: %s', e
                )
                return MessageAction('Error encountered when browsing.')
            set_of_marks = last_obs.set_of_marks
        goal, image_urls = state.get_current_user_intent()

        if goal is None:
            goal = state.inputs['task']
        goal_txt, goal_images = create_goal_prompt(goal, image_urls)
        observation_txt, som_screenshot = create_observation_prompt(
            cur_axtree_txt, tabs, focused_element, error_prefix, set_of_marks
        )
        human_prompt: list[TextContent | ImageContent] = [
            TextContent(type='text', text=goal_txt)
        ]
        if len(goal_images) > 0:
            human_prompt.append(ImageContent(image_urls=goal_images))
        human_prompt.append(TextContent(type='text', text=observation_txt))
        if som_screenshot is not None:
            human_prompt.append(ImageContent(image_urls=[som_screenshot]))
        remaining_content = f"""
{history_prompt}\
{self.action_prompt}\
{self.hints}\
{self.abstract_example}\
{self.concrete_example}\
"""
        human_prompt.append(TextContent(type='text', text=remaining_content))

        system_msg = """\
You are an agent trying to solve a web task based on the content of the page and user instructions. You can interact with the page and explore, and send messages to the user when you finish the task. Each time you submit an action it will be sent to the browser and you will receive a new page.
""".strip()

        messages.append(Message(role='system', content=[TextContent(text=system_msg)]))
        messages.append(Message(role='user', content=human_prompt))

        response = self.llm.completion(
            messages=messages,
            temperature=0.0,
            stop=[')```', ')\n```'],
        )

        return self.response_parser.parse(response)
