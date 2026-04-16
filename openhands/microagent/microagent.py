import io
import re
from itertools import chain
from pathlib import Path
from typing import ClassVar, Union

import frontmatter
from pydantic import BaseModel

from openhands.core.exceptions import (
    MicroagentValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.microagent.types import InputMetadata, MicroagentMetadata, MicroagentType


class BaseMicroagent(BaseModel):
    """Base class for all microagents."""

    name: str
    content: str
    metadata: MicroagentMetadata
    source: str  # path to the file
    type: MicroagentType

    PATH_TO_THIRD_PARTY_MICROAGENT_NAME: ClassVar[dict[str, str]] = {
        '.cursorrules': 'cursorrules',
        'agents.md': 'agents',
        'agent.md': 'agents',
    }

    @classmethod
    def _handle_third_party(
        cls, path: Path, file_content: str
    ) -> Union['RepoMicroagent', None]:
        # Determine the agent name based on file type
        microagent_name = cls.PATH_TO_THIRD_PARTY_MICROAGENT_NAME.get(path.name.lower())

        # Create RepoMicroagent if we recognized the file type
        if microagent_name is not None:
            return RepoMicroagent(
                name=microagent_name,
                content=file_content,
                metadata=MicroagentMetadata(name=microagent_name),
                source=str(path),
                type=MicroagentType.REPO_KNOWLEDGE,
            )

        return None

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        microagent_dir: Path | None = None,
        file_content: str | None = None,
    ) -> 'BaseMicroagent':
        """Load a microagent from a markdown file with frontmatter.

        The agent's name is derived from its path relative to the microagent_dir.
        """
        path = Path(path) if isinstance(path, str) else path

        # Calculate derived name from relative path if microagent_dir is provided
        # Otherwise, we will rely on the name from metadata later
        derived_name = None
        if microagent_dir is not None:
            # Special handling for files which are not in microagent_dir
            derived_name = cls.PATH_TO_THIRD_PARTY_MICROAGENT_NAME.get(
                path.name.lower()
            ) or str(path.relative_to(microagent_dir).with_suffix(''))

        # Only load directly from path if file_content is not provided
        if file_content is None:
            with open(path) as f:
                file_content = f.read()

        # Legacy repo instructions are stored in .openhands_instructions
        if path.name == '.openhands_instructions':
            return RepoMicroagent(
                name='repo_legacy',
                content=file_content,
                metadata=MicroagentMetadata(name='repo_legacy'),
                source=str(path),
                type=MicroagentType.REPO_KNOWLEDGE,
            )

        # Handle third-party agent instruction files
        third_party_agent = cls._handle_third_party(path, file_content)
        if third_party_agent is not None:
            return third_party_agent

        file_io = io.StringIO(file_content)
        loaded = frontmatter.load(file_io)
        content = loaded.content

        # Handle case where there's no frontmatter or empty frontmatter
        metadata_dict = loaded.metadata or {}

        # Ensure version is always a string (YAML may parse numeric versions as integers)
        if 'version' in metadata_dict and not isinstance(metadata_dict['version'], str):
            metadata_dict['version'] = str(metadata_dict['version'])

        try:
            metadata = MicroagentMetadata(**metadata_dict)

            # Validate MCP tools configuration if present
            if metadata.mcp_tools and metadata.mcp_tools.mcpServers:
                from openhands.core.config.mcp_config import StdioMCPServer

                has_stdio = any(
                    isinstance(s, StdioMCPServer)
                    for s in metadata.mcp_tools.mcpServers.values()
                )
                has_non_stdio = any(
                    not isinstance(s, StdioMCPServer)
                    for s in metadata.mcp_tools.mcpServers.values()
                )
                if has_non_stdio:
                    logger.warning(
                        f'Microagent {metadata.name} has remote servers. Only stdio servers are currently supported.'
                    )
                if not has_stdio:
                    raise MicroagentValidationError(
                        f'Microagent {metadata.name} has MCP tools configuration but no stdio servers. '
                        'Only stdio servers are currently supported.'
                    )
        except Exception as e:
            # Provide more detailed error message for validation errors
            error_msg = f'Error validating microagent metadata in {path.name}: {str(e)}'
            if 'type' in metadata_dict and metadata_dict['type'] not in [
                t.value for t in MicroagentType
            ]:
                valid_types = ', '.join([f'"{t.value}"' for t in MicroagentType])
                error_msg += f'. Invalid "type" value: "{metadata_dict["type"]}". Valid types are: {valid_types}'
            raise MicroagentValidationError(error_msg) from e

        # Create appropriate subclass based on type
        subclass_map = {
            MicroagentType.KNOWLEDGE: KnowledgeMicroagent,
            MicroagentType.REPO_KNOWLEDGE: RepoMicroagent,
            MicroagentType.TASK: TaskMicroagent,
        }

        # Infer the agent type:
        # 1. If inputs exist -> TASK
        # 2. If triggers exist -> KNOWLEDGE
        # 3. Else (no triggers) -> REPO (always active)
        inferred_type: MicroagentType
        if metadata.inputs:
            inferred_type = MicroagentType.TASK
            # Add a trigger for the agent name if not already present
            trigger = f'/{metadata.name}'
            if not metadata.triggers or trigger not in metadata.triggers:
                if not metadata.triggers:
                    metadata.triggers = [trigger]
                else:
                    metadata.triggers.append(trigger)
        elif metadata.triggers:
            inferred_type = MicroagentType.KNOWLEDGE
        else:
            # No triggers, default to REPO
            # This handles cases where 'type' might be missing or defaulted by Pydantic
            inferred_type = MicroagentType.REPO_KNOWLEDGE

        if inferred_type not in subclass_map:
            # This should theoretically not happen with the logic above
            raise ValueError(f'Could not determine microagent type for: {path}')

        # Use derived_name if available (from relative path), otherwise fallback to metadata.name
        agent_name = derived_name if derived_name is not None else metadata.name

        agent_class = subclass_map[inferred_type]
        return agent_class(
            name=agent_name,
            content=content,
            metadata=metadata,
            source=str(path),
            type=inferred_type,
        )


class KnowledgeMicroagent(BaseMicroagent):
    """Knowledge micro-agents provide specialized expertise that's triggered by keywords in conversations.

    They help with:
    - Language best practices
    - Framework guidelines
    - Common patterns
    - Tool usage
    """

    def __init__(self, **data):
        super().__init__(**data)
        if self.type not in [MicroagentType.KNOWLEDGE, MicroagentType.TASK]:
            raise ValueError('KnowledgeMicroagent must have type KNOWLEDGE or TASK')

    def match_trigger(self, message: str) -> str | None:
        """Match a trigger in the message.

        It returns the first trigger that matches the message.
        """
        message = message.lower()
        for trigger in self.triggers:
            if trigger.lower() in message:
                return trigger

        return None

    @property
    def triggers(self) -> list[str]:
        return self.metadata.triggers


class RepoMicroagent(BaseMicroagent):
    """Microagent specialized for repository-specific knowledge and guidelines.

    RepoMicroagents are loaded from `.openhands/microagents/repo.md` files within repositories
    and contain private, repository-specific instructions that are automatically loaded when
    working with that repository. They are ideal for:
        - Repository-specific guidelines
        - Team practices and conventions
        - Project-specific workflows
        - Custom documentation references
    """

    def __init__(self, **data):
        super().__init__(**data)
        if self.type != MicroagentType.REPO_KNOWLEDGE:
            raise ValueError(
                f'RepoMicroagent initialized with incorrect type: {self.type}'
            )


class TaskMicroagent(KnowledgeMicroagent):
    """TaskMicroagent is a special type of KnowledgeMicroagent that requires user input.

    These microagents are triggered by a special format: "/{agent_name}"
    and will prompt the user for any required inputs before proceeding.
    """

    def __init__(self, **data):
        super().__init__(**data)
        if self.type != MicroagentType.TASK:
            raise ValueError(
                f'TaskMicroagent initialized with incorrect type: {self.type}'
            )

        # Append a prompt to ask for missing variables
        self._append_missing_variables_prompt()

    def _append_missing_variables_prompt(self) -> None:
        """Append a prompt to ask for missing variables."""
        # Check if the content contains any variables or has inputs defined
        if not self.requires_user_input() and not self.metadata.inputs:
            return

        prompt = "\n\nIf the user didn't provide any of these variables, ask the user to provide them first before the agent can proceed with the task."
        self.content += prompt

    def extract_variables(self, content: str) -> list[str]:
        """Extract variables from the content.

        Variables are in the format ${variable_name}.
        """
        pattern = r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        matches = re.findall(pattern, content)
        return matches

    def requires_user_input(self) -> bool:
        """Check if this microagent requires user input.

        Returns True if the content contains variables in the format ${variable_name}.
        """
        # Check if the content contains any variables
        variables = self.extract_variables(self.content)
        logger.debug(f'This microagent requires user input: {variables}')
        return len(variables) > 0

    @property
    def inputs(self) -> list[InputMetadata]:
        """Get the inputs for this microagent."""
        return self.metadata.inputs


def load_microagents_from_dir(
    microagent_dir: Union[str, Path],
) -> tuple[dict[str, RepoMicroagent], dict[str, KnowledgeMicroagent]]:
    """Load all microagents from the given directory.

    Note, legacy repo instructions will not be loaded here.

    Args:
        microagent_dir: Path to the microagents directory (e.g. .openhands/microagents)

    Returns:
        Tuple of (repo_agents, knowledge_agents) dictionaries
    """
    if isinstance(microagent_dir, str):
        microagent_dir = Path(microagent_dir)

    repo_agents = {}
    knowledge_agents = {}

    # Load all agents from microagents directory
    logger.debug(f'Loading agents from {microagent_dir}')

    # Always check for .cursorrules and AGENTS.md files in repo root, regardless of whether microagents_dir exists
    special_files = []
    repo_root = microagent_dir.parent.parent

    # Check for .cursorrules
    if (repo_root / '.cursorrules').exists():
        special_files.append(repo_root / '.cursorrules')

    # Check for AGENTS.md (case-insensitive)
    for agents_filename in ['AGENTS.md', 'agents.md', 'AGENT.md', 'agent.md']:
        agents_path = repo_root / agents_filename
        if agents_path.exists():
            special_files.append(agents_path)
            break  # Only add the first one found to avoid duplicates

    # Collect .md files from microagents directory if it exists
    md_files = []
    if microagent_dir.exists():
        md_files = [f for f in microagent_dir.rglob('*.md') if f.name != 'README.md']

    # Process all files in one loop
    for file in chain(special_files, md_files):
        try:
            agent = BaseMicroagent.load(file, microagent_dir)
            if isinstance(agent, RepoMicroagent):
                repo_agents[agent.name] = agent
            elif isinstance(agent, KnowledgeMicroagent):
                # Both KnowledgeMicroagent and TaskMicroagent go into knowledge_agents
                knowledge_agents[agent.name] = agent
        except MicroagentValidationError as e:
            # For validation errors, include the original exception
            error_msg = f'Error loading microagent from {file}: {str(e)}'
            raise MicroagentValidationError(error_msg) from e
        except Exception as e:
            # For other errors, wrap in a ValueError with detailed message
            error_msg = f'Error loading microagent from {file}: {str(e)}'
            raise ValueError(error_msg) from e

    logger.debug(
        f'Loaded {len(repo_agents) + len(knowledge_agents)} microagents: '
        f'{[*repo_agents.keys(), *knowledge_agents.keys()]}'
    )
    return repo_agents, knowledge_agents
