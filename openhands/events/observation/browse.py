from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation

if TYPE_CHECKING:
    from openhands.memory.offloader import ContextOffloader


@dataclass
class BrowserOutputObservation(Observation):
    """This data class represents the output of a browser."""

    url: str
    trigger_by_action: str
    screenshot: str = field(repr=False, default='')  # don't show in repr
    screenshot_path: str | None = field(default=None)  # path to saved screenshot file
    set_of_marks: str = field(default='', repr=False)  # don't show in repr
    error: bool = False
    observation: str = ObservationType.BROWSE
    goal_image_urls: list[str] = field(default_factory=list)
    # do not include in the memory
    open_pages_urls: list[str] = field(default_factory=list)
    active_page_index: int = -1
    dom_object: dict[str, Any] = field(
        default_factory=dict, repr=False
    )  # don't show in repr
    axtree_object: dict[str, Any] = field(
        default_factory=dict, repr=False
    )  # don't show in repr
    extra_element_properties: dict[str, Any] = field(
        default_factory=dict, repr=False
    )  # don't show in repr
    last_browser_action: str = ''
    last_browser_action_error: str = ''
    focused_element_bid: str = ''
    filter_visible_only: bool = False

    # Context offloading fields
    dom_offloaded_path: str | None = field(default=None)
    axtree_offloaded_path: str | None = field(default=None)
    screenshot_thumbnail: str = field(default='', repr=False)  # thumbnail data-URL

    @property
    def message(self) -> str:
        return 'Visited ' + self.url

    def offload_large_content(self, offloader: ContextOffloader) -> None:
        """Offload large browser content to filesystem.

        This method should be called after the observation is created
        but before it's added to the event stream. It offloads:
        - DOM object to JSON file
        - AXTree object to JSON file
        - Screenshot to PNG file (keeps thumbnail for LLM vision)

        IMPORTANT: Even if offloading fails, large objects are cleared from
        memory to prevent memory issues. A warning is logged in this case.

        Args:
            offloader: The context offloader instance.
        """
        if not offloader.enabled:
            return

        # Offload DOM object
        if offloader.config.offload_browser_dom and self.dom_object:
            # Estimate size to determine if we should attempt offload and clear
            # Use repr length as a quick heuristic (faster than json.dumps for size check)
            estimated_size = len(repr(self.dom_object))
            should_clear = estimated_size > offloader.config.max_output_chars

            if should_clear:
                try:
                    result = offloader.offload_json(
                        data=self.dom_object,
                        source_type='browser_dom',
                        type_name='DOM Object',
                    )
                    self.dom_offloaded_path = result.file_path
                    logger.debug(f'Offloaded DOM to {result.file_path}')
                except Exception as e:
                    logger.warning(f'Failed to offload DOM: {e}')
                finally:
                    # Always clear DOM object when it's large, even if offload failed
                    self.dom_object = {}

        # Offload AXTree object
        if offloader.config.offload_browser_axtree and self.axtree_object:
            # Estimate size to determine if we should attempt offload and clear
            estimated_size = len(repr(self.axtree_object))
            should_clear = estimated_size > offloader.config.max_output_chars

            if should_clear:
                try:
                    result = offloader.offload_json(
                        data=self.axtree_object,
                        source_type='browser_axtree',
                        type_name='Accessibility Tree',
                    )
                    self.axtree_offloaded_path = result.file_path
                    logger.debug(f'Offloaded AXTree to {result.file_path}')
                except Exception as e:
                    logger.warning(f'Failed to offload AXTree: {e}')
                finally:
                    # Always clear AXTree object when it's large, even if offload failed
                    self.axtree_object = {}

        # Offload screenshot (keep thumbnail for LLM vision)
        if self.screenshot and offloader.should_offload_bytes(len(self.screenshot)):
            try:
                file_path, thumbnail = offloader.offload_image(
                    base64_data=self.screenshot,
                    source_type='screenshot',
                )
                if file_path:
                    self.screenshot_path = file_path
                    self.screenshot_thumbnail = thumbnail
                    self.screenshot = ''  # Clear full base64
                    logger.debug(f'Offloaded screenshot to {file_path}')
            except Exception as e:
                logger.warning(f'Failed to offload screenshot: {e}')

    def __str__(self) -> str:
        ret = (
            '**BrowserOutputObservation**\n'
            f'URL: {self.url}\n'
            f'Error: {self.error}\n'
            f'Open pages: {self.open_pages_urls}\n'
            f'Active page index: {self.active_page_index}\n'
            f'Last browser action: {self.last_browser_action}\n'
            f'Last browser action error: {self.last_browser_action_error}\n'
            f'Focused element bid: {self.focused_element_bid}\n'
        )
        if self.screenshot_path:
            ret += f'Screenshot saved to: {self.screenshot_path}\n'
        if self.dom_offloaded_path:
            ret += f'DOM offloaded to: {self.dom_offloaded_path}\n'
        if self.axtree_offloaded_path:
            ret += f'AXTree offloaded to: {self.axtree_offloaded_path}\n'
        ret += '--- Agent Observation ---\n'
        ret += self.content
        return ret
