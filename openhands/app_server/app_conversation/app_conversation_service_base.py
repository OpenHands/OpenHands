import logging
import os
import tempfile
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import base62

from openhands.app_server.app_conversation.app_conversation_models import (
    AgentType,
    AppConversationStartTask,
    AppConversationStartTaskStatus,
)
from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
)
from openhands.app_server.app_conversation.skill_loader import (
    load_global_skills,
    load_repo_skills,
    load_sandbox_skills,
    merge_skills,
)
from openhands.app_server.sandbox.sandbox_models import SandboxInfo
from openhands.app_server.user.user_context import UserContext
from openhands.sdk import Agent
from openhands.sdk.context.agent_context import AgentContext
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.skills import load_user_skills
from openhands.sdk.llm import LLM
from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace

_logger = logging.getLogger(__name__)
PRE_COMMIT_HOOK = '.git/hooks/pre-commit'
PRE_COMMIT_LOCAL = '.git/hooks/pre-commit.local'


@dataclass
class AppConversationServiceBase(AppConversationService, ABC):
    """App Conversation service which adds git specific functionality.

    Sets up repositories and installs hooks"""

    init_git_in_empty_workspace: bool
    user_context: UserContext

    async def _load_and_merge_all_skills(
        self,
        sandbox: SandboxInfo,
        remote_workspace: AsyncRemoteWorkspace,
        selected_repository: str | None,
        working_dir: str,
    ) -> list:
        """Load skills from all sources and merge them.

        This method handles all errors gracefully and will return an empty list
        if skill loading fails completely.

        Args:
            remote_workspace: AsyncRemoteWorkspace for loading repo skills
            selected_repository: Repository name or None
            working_dir: Working directory path

        Returns:
            List of merged Skill objects from all sources, or empty list on failure
        """
        try:
            _logger.debug('Loading skills for V1 conversation')

            # Load skills from all sources
            sandbox_skills = load_sandbox_skills(sandbox)
            global_skills = load_global_skills()
            # Load user skills from ~/.openhands/skills/ directory
            # Uses the SDK's load_user_skills() function which handles loading from
            # ~/.openhands/skills/ and ~/.openhands/microagents/ (for backward compatibility)
            try:
                user_skills = load_user_skills()
                _logger.info(
                    f'Loaded {len(user_skills)} user skills: {[s.name for s in user_skills]}'
                )
            except Exception as e:
                _logger.warning(f'Failed to load user skills: {str(e)}')
                user_skills = []
            repo_skills = await load_repo_skills(
                remote_workspace, selected_repository, working_dir
            )

            # Merge all skills (later lists override earlier ones)
            all_skills = merge_skills(
                [sandbox_skills, global_skills, user_skills, repo_skills]
            )

            _logger.info(
                f'Loaded {len(all_skills)} total skills: {[s.name for s in all_skills]}'
            )

            return all_skills
        except Exception as e:
            _logger.warning(f'Failed to load skills: {e}', exc_info=True)
            # Return empty list on failure - skills will be loaded again later if needed
            return []

    def _create_agent_with_skills(self, agent, skills: list):
        """Create or update agent with skills in its context.

        Args:
            agent: The agent to update
            skills: List of Skill objects to add to agent context

        Returns:
            Updated agent with skills in context
        """
        if agent.agent_context:
            # Merge with existing context
            existing_skills = agent.agent_context.skills
            all_skills = merge_skills([skills, existing_skills])
            agent = agent.model_copy(
                update={
                    'agent_context': agent.agent_context.model_copy(
                        update={'skills': all_skills}
                    )
                }
            )
        else:
            # Create new context
            agent_context = AgentContext(skills=skills)
            agent = agent.model_copy(update={'agent_context': agent_context})

        return agent

    async def _load_skills_and_update_agent(
        self,
        sandbox: SandboxInfo,
        agent: Agent,
        remote_workspace: AsyncRemoteWorkspace,
        selected_repository: str | None,
        working_dir: str,
    ):
        """Load all skills and update agent with them.

        Args:
            agent: The agent to update
            remote_workspace: AsyncRemoteWorkspace for loading repo skills
            selected_repository: Repository name or None
            working_dir: Working directory path

        Returns:
            Updated agent with skills loaded into context
        """
        # Load and merge all skills
        all_skills = await self._load_and_merge_all_skills(
            sandbox, remote_workspace, selected_repository, working_dir
        )

        # Update agent with skills
        agent = self._create_agent_with_skills(agent, all_skills)

        return agent

    async def run_setup_scripts(
        self,
        task: AppConversationStartTask,
        sandbox: SandboxInfo,
        workspace: AsyncRemoteWorkspace,
    ) -> AsyncGenerator[AppConversationStartTask, None]:
        task.status = AppConversationStartTaskStatus.PREPARING_REPOSITORY
        yield task
        await self.clone_or_init_git_repo(task, workspace)

        task.status = AppConversationStartTaskStatus.RUNNING_SETUP_SCRIPT
        yield task
        await self.maybe_run_setup_script(workspace)

        task.status = AppConversationStartTaskStatus.SETTING_UP_GIT_HOOKS
        yield task
        await self.maybe_setup_git_hooks(workspace)

        task.status = AppConversationStartTaskStatus.SETTING_UP_SKILLS
        yield task
        await self._load_and_merge_all_skills(
            sandbox,
            workspace,
            task.request.selected_repository,
            workspace.working_dir,
        )

    async def _configure_git_user_settings(
        self,
        workspace: AsyncRemoteWorkspace,
    ) -> None:
        """Configure git global user settings from user preferences.

        Reads git_user_name and git_user_email from user settings and
        configures them as git global settings in the workspace.

        Args:
            workspace: The remote workspace to configure git settings in.
        """
        try:
            user_info = await self.user_context.get_user_info()

            if user_info.git_user_name:
                cmd = f'git config --global user.name "{user_info.git_user_name}"'
                result = await workspace.execute_command(cmd, workspace.working_dir)
                if result.exit_code:
                    _logger.warning(f'Git config user.name failed: {result.stderr}')
                else:
                    _logger.info(
                        f'Git configured with user.name={user_info.git_user_name}'
                    )

            if user_info.git_user_email:
                cmd = f'git config --global user.email "{user_info.git_user_email}"'
                result = await workspace.execute_command(cmd, workspace.working_dir)
                if result.exit_code:
                    _logger.warning(f'Git config user.email failed: {result.stderr}')
                else:
                    _logger.info(
                        f'Git configured with user.email={user_info.git_user_email}'
                    )
        except Exception as e:
            _logger.warning(f'Failed to configure git user settings: {e}')

    async def clone_or_init_git_repo(
        self,
        task: AppConversationStartTask,
        workspace: AsyncRemoteWorkspace,
    ):
        request = task.request

        # Create the projects directory if it does not exist yet
        parent = Path(workspace.working_dir).parent
        result = await workspace.execute_command(
            f'mkdir {workspace.working_dir}', parent
        )
        if result.exit_code:
            _logger.warning(f'mkdir failed: {result.stderr}')

        # Configure git user settings from user preferences
        await self._configure_git_user_settings(workspace)

        if not request.selected_repository:
            if self.init_git_in_empty_workspace:
                _logger.debug('Initializing a new git repository in the workspace.')
                cmd = (
                    'git init && git config --global '
                    f'--add safe.directory {workspace.working_dir}'
                )
                result = await workspace.execute_command(cmd, workspace.working_dir)
                if result.exit_code:
                    _logger.warning(f'Git init failed: {result.stderr}')
            else:
                _logger.info('Not initializing a new git repository.')
            return

        remote_repo_url: str = await self.user_context.get_authenticated_git_url(
            request.selected_repository
        )
        if not remote_repo_url:
            raise ValueError('Missing either Git token or valid repository')

        dir_name = request.selected_repository.split('/')[-1]

        # Clone the repo - this is the slow part!
        clone_command = f'git clone {remote_repo_url} {dir_name}'
        result = await workspace.execute_command(
            clone_command, workspace.working_dir, 120
        )
        if result.exit_code:
            _logger.warning(f'Git clone failed: {result.stderr}')

        # Checkout the appropriate branch
        if request.selected_branch:
            checkout_command = f'git checkout {request.selected_branch}'
        else:
            # Generate a random branch name to avoid conflicts
            random_str = base62.encodebytes(os.urandom(16))
            openhands_workspace_branch = f'openhands-workspace-{random_str}'
            checkout_command = f'git checkout -b {openhands_workspace_branch}'
        git_dir = Path(workspace.working_dir) / dir_name
        result = await workspace.execute_command(checkout_command, git_dir)
        if result.exit_code:
            _logger.warning(f'Git checkout failed: {result.stderr}')

    async def maybe_run_setup_script(
        self,
        workspace: AsyncRemoteWorkspace,
    ):
        """Run .openhands/setup.sh if it exists in the workspace or repository."""
        setup_script = workspace.working_dir + '/.openhands/setup.sh'

        await workspace.execute_command(
            f'chmod +x {setup_script} && source {setup_script}', timeout=600
        )

        # TODO: Does this need to be done?
        # Add the action to the event stream as an ENVIRONMENT event
        # source = EventSource.ENVIRONMENT
        # self.event_stream.add_event(action, source)

    async def maybe_setup_git_hooks(
        self,
        workspace: AsyncRemoteWorkspace,
    ):
        """Set up git hooks if .openhands/pre-commit.sh exists in the workspace or repository."""
        command = 'mkdir -p .git/hooks && chmod +x .openhands/pre-commit.sh'
        result = await workspace.execute_command(command, workspace.working_dir)
        if result.exit_code:
            return

        # Check if there's an existing pre-commit hook
        with tempfile.TemporaryFile(mode='w+t') as temp_file:
            result = workspace.file_download(PRE_COMMIT_HOOK, str(temp_file))
            if result.get('success'):
                _logger.info('Preserving existing pre-commit hook')
                # an existing pre-commit hook exists
                if 'This hook was installed by OpenHands' not in temp_file.read():
                    # Move the existing hook to pre-commit.local
                    command = (
                        f'mv {PRE_COMMIT_HOOK} {PRE_COMMIT_LOCAL} &&'
                        f'chmod +x {PRE_COMMIT_LOCAL}'
                    )
                    result = await workspace.execute_command(
                        command, workspace.working_dir
                    )
                    if result.exit_code != 0:
                        _logger.error(
                            f'Failed to preserve existing pre-commit hook: {result.stderr}',
                        )
                        return

        # write the pre-commit hook
        await workspace.file_upload(
            source_path=Path(__file__).parent / 'git' / 'pre-commit.sh',
            destination_path=PRE_COMMIT_HOOK,
        )

        # Make the pre-commit hook executable
        result = await workspace.execute_command(f'chmod +x {PRE_COMMIT_HOOK}')
        if result.exit_code:
            _logger.error(f'Failed to make pre-commit hook executable: {result.stderr}')
            return

        _logger.info('Git pre-commit hook installed successfully')

    def _create_condenser(
        self,
        llm: LLM,
        agent_type: AgentType,
        condenser_max_size: int | None,
    ) -> LLMSummarizingCondenser:
        """Create a condenser based on user settings and agent type.

        Args:
            llm: The LLM instance to use for condensation
            agent_type: Type of agent (PLAN or DEFAULT)
            condenser_max_size: condenser_max_size setting

        Returns:
            Configured LLMSummarizingCondenser instance
        """
        # LLMSummarizingCondenser has defaults: max_size=120, keep_first=4
        condenser_kwargs = {
            'llm': llm.model_copy(
                update={
                    'usage_id': (
                        'condenser'
                        if agent_type == AgentType.DEFAULT
                        else 'planning_condenser'
                    )
                }
            ),
        }
        # Only override max_size if user has a custom value
        if condenser_max_size is not None:
            condenser_kwargs['max_size'] = condenser_max_size

        condenser = LLMSummarizingCondenser(**condenser_kwargs)

        return condenser
