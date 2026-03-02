# flake8: noqa: E501

# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import asyncio
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
from argparse import Namespace
from typing import Any
from uuid import uuid4

from termcolor import colored

import openhands
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, OpenHandsConfig, SandboxConfig
from openhands.core.config.utils import load_openhands_config
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.event import Event
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.events.stream import EventStreamSubscriber
from openhands.integrations.service_types import ProviderType
from openhands.llm.llm_registry import LLMRegistry
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.interfaces.issue_definitions import (
    ServiceContextIssue,
    ServiceContextPR,
)
from openhands.resolver.issue_handler_factory import IssueHandlerFactory
from openhands.resolver.resolver_output import ResolverOutput
from openhands.resolver.utils import (
    codeact_user_response,
    get_unique_uid,
    identify_token,
    reset_logger_for_multiprocessing,
)
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync

# Don't make this configurable for now, unless we have other competitive agents
AGENT_CLASS = 'CodeActAgent'


class IssueResolver:
    GITLAB_CI = os.getenv('GITLAB_CI') == 'true'

    def __init__(self, args: Namespace) -> None:
        """Initialize the IssueResolver with the given parameters.

        Params initialized:
            owner: Owner of the repo.
            repo: Repository name.
            token: Token to access the repository.
            username: Username to access the repository.
            platform: Platform of the repository.
            runtime_container_image: Container image to use.
            max_iterations: Maximum number of iterations to run.
            output_dir: Output directory to write the results.
            llm_config: Configuration for the language model.
            prompt_template: Prompt template to use.
            issue_type: Type of issue to resolve (issue or pr).
            repo_instruction: Repository instruction to use.
            issue_number: Issue number to resolve.
            comment_id: Optional ID of a specific comment to focus on.
            base_domain: The base domain for the git server.
        """
        parts = args.selected_repo.rsplit('/', 1)
        if len(parts) < 2:
            raise ValueError('Invalid repository format. Expected owner/repo')
        owner, repo = parts

        token = (
            args.token
            or os.getenv('GITHUB_TOKEN')
            or os.getenv('GITLAB_TOKEN')
            or os.getenv('BITBUCKET_TOKEN')
            or os.getenv('AZURE_DEVOPS_TOKEN')
            or os.getenv('FORGEJO_TOKEN')
        )
        username = args.username if args.username else os.getenv('GIT_USERNAME')
        if not username:
            raise ValueError('Username is required.')

        if not token:
            raise ValueError('Token is required.')

        platform = call_async_from_sync(
            identify_token,
            GENERAL_TIMEOUT,
            token,
            args.base_domain,
        )

        repo_instruction = None
        if args.repo_instruction_file:
            with open(args.repo_instruction_file, 'r') as f:
                repo_instruction = f.read()

        issue_type = args.issue_type

        # Read the prompt template
        prompt_file = args.prompt_file
        if prompt_file is None:
            if issue_type == 'issue':
                prompt_file = os.path.join(
                    os.path.dirname(__file__), 'prompts/resolve/basic-with-tests.jinja'
                )
            else:
                prompt_file = os.path.join(
                    os.path.dirname(__file__), 'prompts/resolve/basic-followup.jinja'
                )
        with open(prompt_file, 'r') as f:
            user_instructions_prompt_template = f.read()

        with open(
            prompt_file.replace('.jinja', '-conversation-instructions.jinja')
        ) as f:
            conversation_instructions_prompt_template = f.read()

        base_domain = args.base_domain
        if base_domain is None:
            base_domain = (
                'github.com'
                if platform == ProviderType.GITHUB
                else 'gitlab.com'
                if platform == ProviderType.GITLAB
                else 'bitbucket.org'
                if platform == ProviderType.BITBUCKET
                else 'dev.azure.com'
            )

        self.output_dir = args.output_dir
        self.issue_type = issue_type
        self.issue_number = args.issue_number

        self.workspace_base = self.build_workspace_base(
            self.output_dir, self.issue_type, self.issue_number
        )

        self.max_iterations = args.max_iterations

        self.app_config = self.update_openhands_config(
            load_openhands_config(),
            self.max_iterations,
            self.workspace_base,
            args.base_container_image,
            args.runtime_container_image,
            args.is_experimental,
            args.runtime,
        )

        self.owner = owner
        self.repo = repo
        self.platform = platform
        self.user_instructions_prompt_template = user_instructions_prompt_template
        self.conversation_instructions_prompt_template = (
            conversation_instructions_prompt_template
        )
        self.repo_instruction = repo_instruction
        self.comment_id = args.comment_id

        factory = IssueHandlerFactory(
            owner=self.owner,
            repo=self.repo,
            token=token,
            username=username,
            platform=self.platform,
            base_domain=base_domain,
            issue_type=self.issue_type,
            llm_config=self.app_config.get_llm_config(),
        )
        self.issue_handler = factory.create()

    @classmethod
    def update_openhands_config(
        cls,
        config: OpenHandsConfig,
        max_iterations: int,
        workspace_base: str,
        base_container_image: str | None,
        runtime_container_image: str | None,
        is_experimental: bool,
        runtime: str | None = None,
    ) -> OpenHandsConfig:
        config.default_agent = 'CodeActAgent'
        # Use provided runtime or fallback to config value or default to 'docker'
        config.runtime = runtime or config.runtime or 'docker'
        config.max_budget_per_task = 4
        config.max_iterations = max_iterations

        # do not mount workspace
        config.workspace_base = workspace_base
        config.workspace_mount_path = workspace_base
        config.agents = {'CodeActAgent': AgentConfig(disabled_microagents=['github'])}

        cls.update_sandbox_config(
            config,
            base_container_image,
            runtime_container_image,
            is_experimental,
        )

        return config

    @classmethod
    def update_sandbox_config(
        cls,
        openhands_config: OpenHandsConfig,
        base_container_image: str | None,
        runtime_container_image: str | None,
        is_experimental: bool,
    ) -> None:
        if runtime_container_image is not None and base_container_image is not None:
            raise ValueError('Cannot provide both runtime and base container images.')

        if (
            runtime_container_image is None
            and base_container_image is None
            and not is_experimental
        ):
            runtime_container_image = (
                f'ghcr.io/openhands/runtime:{openhands.__version__}-nikolaik'
            )

        # Convert container image values to string or None
        container_base = (
            str(base_container_image) if base_container_image is not None else None
        )
        container_runtime = (
            str(runtime_container_image)
            if runtime_container_image is not None
            else None
        )

        sandbox_config = SandboxConfig(
            base_container_image=container_base,
            runtime_container_image=container_runtime,
            enable_auto_lint=False,
            use_host_network=False,
            timeout=300,
        )

        # Configure sandbox for GitLab CI environment
        if cls.GITLAB_CI:
            sandbox_config.local_runtime_url = os.getenv(
                'LOCAL_RUNTIME_URL', 'http://localhost'
            )
            user_id = os.getuid() if hasattr(os, 'getuid') else 1000
            if user_id == 0:
                sandbox_config.user_id = get_unique_uid()

        openhands_config.sandbox.base_container_image = (
            sandbox_config.base_container_image
        )
        openhands_config.sandbox.runtime_container_image = (
            sandbox_config.runtime_container_image
        )
        openhands_config.sandbox.enable_auto_lint = sandbox_config.enable_auto_lint
        openhands_config.sandbox.use_host_network = sandbox_config.use_host_network
        openhands_config.sandbox.timeout = sandbox_config.timeout
        openhands_config.sandbox.local_runtime_url = sandbox_config.local_runtime_url
        openhands_config.sandbox.user_id = sandbox_config.user_id

    def initialize_runtime(
        self,
        runtime: Runtime,
    ) -> None:
        """Initialize the runtime for the agent.

        This function is called before the runtime is used to run the agent.
        It sets up git configuration and runs the setup script if it exists.
        """
        logger.info('-' * 30)
        logger.info('BEGIN Runtime Completion Fn')
        logger.info('-' * 30)
        obs: Observation

        action = CmdRunAction(command='cd /workspace')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to change directory to /workspace.\n{obs}')

        if self.platform == ProviderType.GITLAB and self.GITLAB_CI:
            action = CmdRunAction(command='sudo chown -R 1001:0 /workspace/*')
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action = CmdRunAction(command='git config --global core.pager ""')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to set git config.\n{obs}')

        # Run setup script if it exists
        logger.info('Checking for .openhands/setup.sh script...')
        runtime.maybe_run_setup_script()

        # Setup git hooks if they exist
        logger.info('Checking for .openhands/pre-commit.sh script...')
        runtime.maybe_setup_git_hooks()

    async def complete_runtime(
        self,
        runtime: Runtime,
        base_commit: str,
    ) -> dict[str, Any]:
        """Complete the runtime for the agent.

        This function is called before the runtime is used to run the agent.
        If you need to do something in the sandbox to get the correctness metric after
        the agent has run, modify this function.
        """
        logger.info('-' * 30)
        logger.info('BEGIN Runtime Completion Fn')
        logger.info('-' * 30)
        obs: Observation

        action = CmdRunAction(command='cd /workspace')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(
                f'Failed to change directory to /workspace. Observation: {obs}'
            )

        action = CmdRunAction(command='git config --global core.pager ""')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to set git config. Observation: {obs}')

        action = CmdRunAction(
            command='git config --global --add safe.directory /workspace'
        )
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to set git config. Observation: {obs}')

        if self.platform == ProviderType.GITLAB and self.GITLAB_CI:
            action = CmdRunAction(command='sudo git add -A')
        else:
            action = CmdRunAction(command='git add -A')

        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to git add. Observation: {obs}')

        n_retries = 0
        git_patch = None
        while n_retries < 5:
            action = CmdRunAction(command=f'git diff --no-color --cached {base_commit}')
            action.set_hard_timeout(600 + 100 * n_retries)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            n_retries += 1
            if isinstance(obs, CmdOutputObservation):
                if obs.exit_code == 0:
                    git_patch = obs.content.strip()
                    break
                else:
                    logger.info('Failed to get git diff, retrying...')
                    await asyncio.sleep(10)
            elif isinstance(obs, ErrorObservation):
                logger.error(f'Error occurred: {obs.content}. Retrying...')
                await asyncio.sleep(10)
            else:
                raise ValueError(f'Unexpected observation type: {type(obs)}')

        logger.info('-' * 30)
        logger.info('END Runtime Completion Fn')
        logger.info('-' * 30)
        return {'git_patch': git_patch}

    @staticmethod
    def build_workspace_base(
        output_dir: str, issue_type: str, issue_number: int
    ) -> str:
        workspace_base = os.path.join(
            output_dir, 'workspace', f'{issue_type}_{issue_number}'
        )
        return os.path.abspath(workspace_base)

    async def process_issue(
        self,
        issue: Issue,
        base_commit: str,
        issue_handler: ServiceContextIssue | ServiceContextPR,
        reset_logger: bool = False,
    ) -> ResolverOutput:
        # Setup the logger properly, so you can run multi-processing to parallelize processing
        if reset_logger:
            log_dir = os.path.join(self.output_dir, 'infer_logs')
            reset_logger_for_multiprocessing(logger, str(issue.number), log_dir)
        else:
            logger.info(f'Starting fixing issue {issue.number}.')

        # write the repo to the workspace
        if os.path.exists(self.workspace_base):
            shutil.rmtree(self.workspace_base)
        shutil.copytree(os.path.join(self.output_dir, 'repo'), self.workspace_base)

        llm_registry = LLMRegistry(self.app_config)
        runtime = create_runtime(self.app_config, llm_registry)
        await runtime.connect()

        def on_event(evt: Event) -> None:
            logger.info(evt)

        runtime.event_stream.subscribe(
            EventStreamSubscriber.MAIN, on_event, str(uuid4())
        )

        self.initialize_runtime(runtime)

        instruction, conversation_instructions, images_urls = (
            issue_handler.get_instruction(
                issue,
                self.user_instructions_prompt_template,
                self.conversation_instructions_prompt_template,
                self.repo_instruction,
            )
        )
        # Here's how you can run the agent (similar to the `main` function) and get the final task state
        action = MessageAction(content=instruction, image_urls=images_urls)
        try:
            state: State | None = await run_controller(
                config=self.app_config,
                initial_user_action=action,
                runtime=runtime,
                fake_user_response_fn=codeact_user_response,
                conversation_instructions=conversation_instructions,
            )
            if state is None:
                raise RuntimeError('Failed to run the agent.')
        except (ValueError, RuntimeError) as e:
            error_msg = f'Agent failed with error: {str(e)}'
            logger.error(error_msg)
            state = None
            last_error: str | None = error_msg

        # Get git patch
        return_val = await self.complete_runtime(runtime, base_commit)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {issue.number}:\n--------\n{git_patch}\n--------'
        )

        # Serialize histories and set defaults for failed state
        if state is None:
            histories = []
            metrics = None
            success = False
            comment_success = None
            result_explanation = 'Agent failed to run'
            last_error = 'Agent failed to run or crashed'
        else:
            histories = [dataclasses.asdict(event) for event in state.history]
            metrics = state.metrics.get() if state.metrics else None
            # determine success based on the history, issue description and git patch
            success, comment_success, result_explanation = issue_handler.guess_success(
                issue, state.history, git_patch
            )

            if issue_handler.issue_type == 'pr' and comment_success:
                success_log = 'I have updated the PR and resolved some of the issues that were cited in the pull request review. Specifically, I identified the following revision requests, and all the ones that I think I successfully resolved are checked off. All the unchecked ones I was not able to resolve, so manual intervention may be required:\n'
                try:
                    explanations = json.loads(result_explanation)
                except json.JSONDecodeError:
                    logger.error(
                        f'Failed to parse result_explanation as JSON: {result_explanation}'
                    )
                    explanations = [
                        str(result_explanation)
                    ]  # Use raw string as fallback

                for success_indicator, explanation in zip(
                    comment_success, explanations
                ):
                    status = (
                        colored('[X]', 'red')
                        if success_indicator
                        else colored('[ ]', 'red')
                    )
                    bullet_point = colored('-', 'yellow')
                    success_log += f'\n{bullet_point} {status}: {explanation}'
                logger.info(success_log)
            last_error = state.last_error if state.last_error else None

        # Save the output
        output = ResolverOutput(
            issue=issue,
            issue_type=issue_handler.issue_type,
            instruction=instruction,
            base_commit=base_commit,
            git_patch=git_patch,
            history=histories,
            metrics=metrics,
            success=success,
            comment_success=comment_success,
            result_explanation=result_explanation,
            error=last_error,
        )
        return output

    def extract_issue(self) -> Issue:
        # Load dataset
        issues: list[Issue] = self.issue_handler.get_converted_issues(
            issue_numbers=[self.issue_number], comment_id=self.comment_id
        )

        if not issues:
            raise ValueError(
                f'No issues found for issue number {self.issue_number}. Please verify that:\n'
                f'1. The issue/PR #{self.issue_number} exists in the repository {self.owner}/{self.repo}\n'
                f'2. You have the correct permissions to access it\n'
                f'3. The repository name is spelled correctly'
            )

        return issues[0]

    async def resolve_issue(
        self,
        reset_logger: bool = False,
    ) -> None:
        """Resolve a single issue.

        Args:
            reset_logger: Whether to reset the logger for multiprocessing.
        """
        issue = self.extract_issue()

        if self.comment_id is not None:
            if (
                self.issue_type == 'pr'
                and not issue.review_comments
                and not issue.review_threads
                and not issue.thread_comments
            ):
                raise ValueError(
                    f'Comment ID {self.comment_id} did not have a match for issue {issue.number}'
                )

            if self.issue_type == 'issue' and not issue.thread_comments:
                raise ValueError(
                    f'Comment ID {self.comment_id} did not have a match for issue {issue.number}'
                )

        # TEST METADATA
        model_name = self.app_config.get_llm_config().model.split('/')[-1]

        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.output_dir, 'infer_logs')).mkdir(
            parents=True, exist_ok=True
        )
        logger.info(f'Using output directory: {self.output_dir}')

        # checkout the repo
        repo_dir = os.path.join(self.output_dir, 'repo')
        if not os.path.exists(repo_dir):
            checkout_output = subprocess.check_output(
                [
                    'git',
                    'clone',
                    self.issue_handler.get_clone_url(),
                    f'{self.output_dir}/repo',
                ]
            ).decode('utf-8')
            if 'fatal' in checkout_output:
                raise RuntimeError(f'Failed to clone repository: {checkout_output}')

        # get the commit id of current repo for reproducibility
        base_commit = (
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir)
            .decode('utf-8')
            .strip()
        )
        logger.info(f'Base commit: {base_commit}')

        if self.repo_instruction is None:
            # Check for .openhands_instructions file in the workspace directory
            openhands_instructions_path = os.path.join(
                repo_dir, '.openhands_instructions'
            )
            if os.path.exists(openhands_instructions_path):
                with open(openhands_instructions_path, 'r') as f:
                    self.repo_instruction = f.read()

        # OUTPUT FILE
        output_file = os.path.join(self.output_dir, 'output.jsonl')
        logger.info(f'Writing output to {output_file}')

        # Check if this issue was already processed
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    data = ResolverOutput.model_validate_json(line)
                    if data.issue.number == self.issue_number:
                        logger.warning(
                            f'Issue {self.issue_number} was already processed. Skipping.'
                        )
                        return

        output_fp = open(output_file, 'a')

        logger.info(
            f'Resolving issue {self.issue_number} with Agent {AGENT_CLASS}, model {model_name}, max iterations {self.max_iterations}.'
        )

        try:
            # checkout to pr branch if needed
            if self.issue_type == 'pr':
                branch_to_use = issue.head_branch
                logger.info(
                    f'Checking out to PR branch {branch_to_use} for issue {issue.number}'
                )

                if not branch_to_use:
                    raise ValueError('Branch name cannot be None')

                # Fetch the branch first to ensure it exists locally
                fetch_cmd = ['git', 'fetch', 'origin', branch_to_use]
                subprocess.check_output(
                    fetch_cmd,
                    cwd=repo_dir,
                )

                # Checkout the branch
                checkout_cmd = ['git', 'checkout', branch_to_use]
                subprocess.check_output(
                    checkout_cmd,
                    cwd=repo_dir,
                )

                base_commit = (
                    subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir)
                    .decode('utf-8')
                    .strip()
                )

            output = await self.process_issue(
                issue,
                base_commit,
                self.issue_handler,
                reset_logger,
            )
            output_fp.write(output.model_dump_json() + '\n')
            output_fp.flush()

        finally:
            output_fp.close()
            logger.info('Finished.')
