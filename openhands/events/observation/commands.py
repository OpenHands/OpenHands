from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation

if TYPE_CHECKING:
    from openhands.memory.offloader import ContextOffloader

CMD_OUTPUT_PS1_BEGIN = '\n###PS1JSON###\n'
CMD_OUTPUT_PS1_END = '\n###PS1END###'
CMD_OUTPUT_METADATA_PS1_REGEX = re.compile(
    f'^{CMD_OUTPUT_PS1_BEGIN.strip()}(.*?){CMD_OUTPUT_PS1_END.strip()}',
    re.DOTALL | re.MULTILINE,
)

# Default max size for command output content
# to prevent too large observations from being saved in the stream
# This matches the default max_message_chars in LLMConfig
MAX_CMD_OUTPUT_SIZE: int = 30000


class CmdOutputMetadata(BaseModel):
    """Additional metadata captured from PS1"""

    exit_code: int = -1
    pid: int = -1
    username: str | None = None
    hostname: str | None = None
    working_dir: str | None = None
    py_interpreter_path: str | None = None
    prefix: str = ''  # Prefix to add to command output
    suffix: str = ''  # Suffix to add to command output

    @classmethod
    def to_ps1_prompt(cls) -> str:
        """Convert the required metadata into a PS1 prompt."""
        prompt = CMD_OUTPUT_PS1_BEGIN
        json_str = json.dumps(
            {
                'pid': '$!',
                'exit_code': '$?',
                'username': r'\u',
                'hostname': r'\h',
                'working_dir': r'$(pwd)',
                'py_interpreter_path': r'$(which python 2>/dev/null || echo "")',
            },
            indent=2,
        )
        # Make sure we escape double quotes in the JSON string
        # So that PS1 will keep them as part of the output
        prompt += json_str.replace('"', r'\"')
        prompt += CMD_OUTPUT_PS1_END + '\n'  # Ensure there's a newline at the end
        return prompt

    @classmethod
    def matches_ps1_metadata(cls, string: str) -> list[re.Match[str]]:
        matches = []
        for match in CMD_OUTPUT_METADATA_PS1_REGEX.finditer(string):
            try:
                json.loads(match.group(1).strip())  # Try to parse as JSON
                matches.append(match)
            except json.JSONDecodeError:
                logger.warning(
                    f'Failed to parse PS1 metadata: {match.group(1)}. Skipping.',
                    exc_info=True,
                )
                continue  # Skip if not valid JSON
        return matches

    @classmethod
    def from_ps1_match(cls, match: re.Match[str]) -> Self:
        """Extract the required metadata from a PS1 prompt."""
        metadata = json.loads(match.group(1))
        # Create a copy of metadata to avoid modifying the original
        processed = metadata.copy()
        # Convert numeric fields
        if 'pid' in metadata:
            try:
                processed['pid'] = int(float(str(metadata['pid'])))
            except (ValueError, TypeError):
                processed['pid'] = -1
        if 'exit_code' in metadata:
            try:
                processed['exit_code'] = int(float(str(metadata['exit_code'])))
            except (ValueError, TypeError):
                logger.warning(
                    f'Failed to parse exit code: {metadata["exit_code"]}. Setting to -1.'
                )
                processed['exit_code'] = -1
        return cls(**processed)


@dataclass
class CmdOutputObservation(Observation):
    """This data class represents the output of a command."""

    command: str
    observation: str = ObservationType.RUN
    # Additional metadata captured from PS1
    metadata: CmdOutputMetadata = field(default_factory=CmdOutputMetadata)
    # Whether the command output should be hidden from the user
    hidden: bool = False
    # Context offloading fields
    offloaded_path: str | None = field(default=None)
    original_size: int | None = field(default=None)

    def __init__(
        self,
        content: str,
        command: str,
        observation: str = ObservationType.RUN,
        metadata: dict[str, Any] | CmdOutputMetadata | None = None,
        hidden: bool = False,
        offloader: ContextOffloader | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize offload fields
        self.offloaded_path = kwargs.pop('offloaded_path', None)
        self.original_size = kwargs.pop('original_size', None)

        # Process content: try offload first, then truncate as fallback
        # Hidden commands don't go through LLM/event stream, so no need to process
        if not hidden:
            content = self._process_content(content, offloader)

        super().__init__(content)

        self.command = command
        self.observation = observation
        self.hidden = hidden
        if isinstance(metadata, dict):
            self.metadata = CmdOutputMetadata(**metadata)
        else:
            self.metadata = metadata or CmdOutputMetadata()

        # Handle legacy attribute
        if 'exit_code' in kwargs:
            self.metadata.exit_code = kwargs['exit_code']
        if 'command_id' in kwargs:
            self.metadata.pid = kwargs['command_id']

    def _process_content(
        self,
        content: str,
        offloader: ContextOffloader | None,
    ) -> str:
        """Process content: try offload first (lossless), then truncate (lossy).

        Args:
            content: The raw content to process.
            offloader: Optional context offloader for saving large outputs.

        Returns:
            Processed content (either preview message or truncated content).
        """
        # Try offloading first (lossless - preserves full content in file)
        if offloader is not None and offloader.should_offload(content):
            try:
                result = offloader.offload_text(
                    content=content,
                    source_type='cmd',
                )
                self.offloaded_path = result.file_path
                self.original_size = result.original_size
                return result.preview_message
            except Exception as e:
                # Offload failed, fall back to truncation
                logger.warning(f'Offload failed, falling back to truncation: {e}')

        # Fallback: truncate (lossy - loses middle content)
        return self._maybe_truncate(content)

    @staticmethod
    def _maybe_truncate(content: str, max_size: int = MAX_CMD_OUTPUT_SIZE) -> str:
        """Truncate the content if it's too large.

        This helps avoid storing unnecessarily large content in the event stream.

        Args:
            content: The content to truncate
            max_size: Maximum size before truncation. Defaults to MAX_CMD_OUTPUT_SIZE.

        Returns:
            Original content if not too large, or truncated content otherwise
        """
        if len(content) <= max_size:
            return content

        # Truncate the middle and include a message about it
        half = max_size // 2
        original_length = len(content)
        truncated = (
            content[:half]
            + '\n[... Observation truncated due to length ...]\n'
            + content[-half:]
        )
        logger.debug(
            f'Truncated large command output: {original_length} -> {len(truncated)} chars'
        )
        return truncated

    @property
    def command_id(self) -> int:
        return self.metadata.pid

    @property
    def exit_code(self) -> int:
        return self.metadata.exit_code

    @property
    def error(self) -> bool:
        return self.exit_code != 0

    @property
    def message(self) -> str:
        return f'Command `{self.command}` executed with exit code {self.exit_code}.'

    @property
    def success(self) -> bool:
        return not self.error

    def __str__(self) -> str:
        return (
            f'**CmdOutputObservation (source={self.source}, exit code={self.exit_code}, '
            f'metadata={json.dumps(self.metadata.model_dump(), indent=2)})**\n'
            '--BEGIN AGENT OBSERVATION--\n'
            f'{self.to_agent_observation()}\n'
            '--END AGENT OBSERVATION--'
        )

    def to_agent_observation(self) -> str:
        ret = f'{self.metadata.prefix}{self.content}{self.metadata.suffix}'
        if self.metadata.working_dir:
            ret += f'\n[Current working directory: {self.metadata.working_dir}]'
        if self.metadata.py_interpreter_path:
            ret += f'\n[Python interpreter: {self.metadata.py_interpreter_path}]'
        if self.metadata.exit_code != -1:
            ret += f'\n[Command finished with exit code {self.metadata.exit_code}]'
        return ret


@dataclass
class IPythonRunCellObservation(Observation):
    """This data class represents the output of a IPythonRunCellAction."""

    code: str
    observation: str = ObservationType.RUN_IPYTHON
    image_urls: list[str] | None = None
    # Context offloading fields
    offloaded_path: str | None = field(default=None)
    original_size: int | None = field(default=None)

    def offload_content(self, offloader: ContextOffloader) -> None:
        """Offload large content to filesystem if needed.

        Args:
            offloader: The context offloader instance.
        """
        if not offloader.enabled:
            return

        if offloader.should_offload(self.content):
            try:
                result = offloader.offload_text(
                    content=self.content,
                    source_type='ipython',
                )
                self.offloaded_path = result.file_path
                self.original_size = result.original_size
                self.content = result.preview_message
            except Exception as e:
                logger.warning(f'IPython offload failed: {e}')

    @property
    def error(self) -> bool:
        return False  # IPython cells do not return exit codes

    @property
    def message(self) -> str:
        return 'Code executed in IPython cell.'

    @property
    def success(self) -> bool:
        return True  # IPython cells are always considered successful

    def __str__(self) -> str:
        result = f'**IPythonRunCellObservation**\n{self.content}'
        if self.image_urls:
            result += f'\nImages: {len(self.image_urls)}'
        if self.offloaded_path:
            result += f'\nOffloaded to: {self.offloaded_path}'
        return result
