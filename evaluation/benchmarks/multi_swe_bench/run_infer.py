import asyncio
import json
import os
import tempfile
from typing import Any

import pandas as pd
import toml
from datasets import load_dataset

import openhands.agenthub
from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    check_maximum_retries_exceeded,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
    is_fatal_evaluation_error,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    get_evaluation_parser,
    get_llm_config_arg,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.events.serialization.event import event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
USE_INSTANCE_IMAGE = os.environ.get('USE_INSTANCE_IMAGE', 'true').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'

# TODO: migrate all swe-bench docker to ghcr.io/openhands
# TODO: 适应所有的语言
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'mswebench')
LANGUAGE = os.environ.get('LANGUAGE', 'java')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    return f'{instance.repo}__{instance.version}'.replace('/', '__')


def get_instruction(instance: pd.Series, metadata: EvalMetadata):
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    # Prepare instruction

    # Instruction based on Anthropic's official trajectory
    # https://github.com/eschluntz/swe-bench-experiments/tree/main/evaluation/verified/20241022_tools_claude-3-5-sonnet-updated/trajs
    instructions = {
        'python': (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following issue description:\n\n"
            f'<issue_description>\n'
            f'{instance.problem_statement}\n'
            '</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {instance["base_commit"]}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'java': (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a Java code repository in the directory {workspace_dir_name}. Consider the following issue description:\n\n"
            f'<issue_description>\n'
            f'{instance.problem_statement}\n'
            '</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Java environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a Java class to reproduce the error and execute it by first compiling with `javac <classname>.java` and then running with `java <classname>` using the BashTool, to confirm the error\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script or class and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce class or script, and run them to make sure your fix handles these cases as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {instance["base_commit"]}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions or classes you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'go': (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a Go code repository in the directory {workspace_dir_name}. Consider the following issue description:\n\n"
            f'<issue_description>\n'
            f'{instance.problem_statement}\n'
            '</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Go environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script or a function to reproduce the error and execute it with `go run <filename.go>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {instance["base_commit"]}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'c': (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a C code repository in the directory {workspace_dir_name}. Consider the following issue description:\n\n"
            f'<issue_description>\n'
            f'{instance.problem_statement}\n'
            '</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development C environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error by compiling your C code (for example, using `gcc <filename.c> -o <executable>`) and then running the executable using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {instance["base_commit"]}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'cpp': (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a C++ code repository in the directory {workspace_dir_name}. Consider the following issue description:\n\n"
            f'<issue_description>\n'
            f'{instance.problem_statement}\n'
            '</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development C++ environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create or adapt a small executable (e.g., a main file or a test driver) to reproduce the issue. Build and run it (for example, by using `g++ -o reproduce reproduce.cpp && ./reproduce` via the BashTool) to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {instance["base_commit"]}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'javascript': (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a Javascript code repository in the directory {workspace_dir_name}. Consider the following issue description:\n\n"
            f'<issue_description>\n'
            f'{instance.problem_statement}\n'
            '</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Javascript environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error and execute it with `node <filename.js>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {instance["base_commit"]}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'typescript': (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a Typescript code repository in the directory {workspace_dir_name}. Consider the following issue description:\n\n"
            f'<issue_description>\n'
            f'{instance.problem_statement}\n'
            '</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Typescript environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error and execute it with `ts-node <filename.ts>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {instance["base_commit"]}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'rust': (
            '<uploaded_files>\n'
            f'/workspace/{workspace_dir_name}\n'
            '</uploaded_files>\n'
            f"I've uploaded a Rust code repository in the directory {workspace_dir_name}. Consider the following issue description:\n\n"
            f'<issue_description>\n'
            f'{instance.problem_statement}\n'
            '</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Rust environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a reproduction script (or binary) that triggers the error and execute it with `cargo run --bin <filename>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {instance["base_commit"]}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
    }
    instruction = instructions.get(LANGUAGE.lower())

    if instruction and RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
        )
    return instruction


# TODO: 适应所有的语言
# def get_instance_docker_image(instance_id: str) -> str:
#     image_name = 'sweb.eval.x86_64.' + instance_id
#     if LANGUAGE == 'python':
#         image_name = image_name.replace(
#             '__', '_s_'
#         )  # to comply with docker image naming convention
#         return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()
#     else:
#         return image_name.lower() ##加载本地的
def get_instance_docker_image(instance: pd.Series):
    if LANGUAGE == 'python':
        image_name = 'sweb.eval.x86_64.' + instance['instance_id']
        image_name = image_name.replace(
            '__', '_s_'
        )  # to comply with docker image naming convention
        return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()
    else:
        container_name = instance.get('repo', '').lower()
        container_name = container_name.replace('/', '_m_')
        instance_id = instance.get('instance_id', '')
        tag_suffix = instance_id.split('-')[-1] if instance_id else ''
        container_tag = f'pr-{tag_suffix}'
        return f'{DOCKER_IMAGE_PREFIX}/{container_name}:{container_tag}'


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    base_container_image = get_instance_docker_image(instance)
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/OpenHands/OpenHands if you run into any issues.'
    )

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )

    config = get_openhands_config_for_eval(
        metadata=metadata,
        enable_browser=RUN_WITH_BROWSING,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
    )
    config.set_agent_config(agent_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    obs: CmdOutputObservation

    REPO_NAME = instance['repo'].split('/')[-1]
    # Set instance id
    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc && echo 'export REPO_NAME={REPO_NAME}' >> ~/.bashrc"""
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0, f'Failed to export SWE_INSTANCE_ID: {str(obs)}'
    )
    # pdb.set_trace()
    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')

    if USE_INSTANCE_IMAGE:
        # inject the init script
        script_dir = os.path.dirname(__file__)

        # inject the instance info
        action = CmdRunAction(command='mkdir -p /swe_util/eval_data/instances')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to create /swe_util/eval_data/instances: {str(obs)}',
        )

        swe_instance_json_name = 'swe-bench-instance.json'
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construct the full path for the desired file name within the temporary directory
            temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
            # Write to the file with the desired name within the temporary directory
            with open(temp_file_path, 'w') as f:
                if not isinstance(instance, dict):
                    json.dump([instance.to_dict()], f)
                else:
                    json.dump([instance], f)

            # Copy the file to the desired location
            runtime.copy_to(temp_file_path, '/swe_util/eval_data/instances/')

        # inject the instance swe entry
        runtime.copy_to(
            str(os.path.join(script_dir, 'scripts/setup/instance_swe_entry.sh')),
            '/swe_util/',
        )
        action = CmdRunAction(command='cat ~/.bashrc')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to cat ~/.bashrc: {str(obs)}')

        action = CmdRunAction(command='source ~/.bashrc')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if isinstance(obs, ErrorObservation):
            logger.error(f'Failed to source ~/.bashrc: {str(obs)}')
        assert_and_raise(obs.exit_code == 0, f'Failed to source ~/.bashrc: {str(obs)}')

        action = CmdRunAction(command='source /swe_util/instance_swe_entry.sh')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to source /swe_util/instance_swe_entry.sh: {str(obs)}',
        )
    else:
        action = CmdRunAction(command='source /swe_util/swe_entry.sh')
        action.set_hard_timeout(1800)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to source /swe_util/swe_entry.sh: {str(obs)}',
        )

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    action = CmdRunAction(command='git reset --hard')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

    action = CmdRunAction(
        command='for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to remove git remotes: {str(obs)}')
    ##TODO:这里看看需不需要判断其他语言的环境
    # action = CmdRunAction(command='which python')
    # action.set_hard_timeout(600)
    # logger.info(action, extra={'msg_type': 'ACTION'})
    # obs = runtime.run_action(action)
    # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    # assert_and_raise(
    #     obs.exit_code == 0 and 'testbed' in obs.content,
    #     f'Expected to find python interpreter from testbed, but got: {str(obs)}',
    # )

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required, but it is used to get the workspace_dir_name
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        # The previous command is still running
        # We need to kill previous command
        logger.info('The previous command is still running, trying to kill it...')
        action = CmdRunAction(command='C-c')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Then run the command again
        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    action = CmdRunAction(command='git config --global core.pager ""')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git config --global core.pager "": {str(obs)}',
    )

    action = CmdRunAction(command='git add -A')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git add -A: {str(obs)}',
    )

    ##删除二进制文件
    action = CmdRunAction(
        command="""
        for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
            if [ -f "$file" ] && (file "$file" | grep -q "executable" || git check-attr binary "$file" | grep -q "binary: set"); then
                git rm -f "$file" 2>/dev/null || rm -f "$file"
                echo "Removed: $file"
            fi
        done
        """
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to remove binary files: {str(obs)}',
    )

    # pdb.set_trace()

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {instance["base_commit"]} > patch.diff'
        )
        action.set_hard_timeout(max(300 + 100 * n_retries, 600))
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                # git_patch = obs.content.strip()
                break
            else:
                logger.info('Failed to get git diff, retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
        else:
            assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    action = FileReadAction(path='patch.diff')
    action.set_hard_timeout(max(300 + 100 * n_retries, 600))
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    git_patch = obs.content
    # pdb.set_trace()

    assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {'git_patch': git_patch}


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    config = get_config(instance, metadata)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    # Increase resource_factor with increasing attempt_id
    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )
    # pdb.set_trace()
    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    try:
        initialize_runtime(runtime, instance)

        instruction = get_instruction(instance, metadata)

        # Here's how you can run the agent (similar to the `main` function) and get the final task state
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=MessageAction(content=instruction),
                runtime=runtime,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                    metadata.agent_class
                ],
            )
        )

        # if fatal error, throw EvalError to trigger re-run
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

        # ======= THIS IS SWE-Bench specific =======
        # Get git patch
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
    finally:
        runtime.close()
    # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    ###remove binary diffs
    def remove_binary_diffs(patch_text):
        lines = patch_text.splitlines()
        cleaned_lines = []
        block = []
        is_binary_block = False

        for line in lines:
            if line.startswith('diff --git '):
                if block and not is_binary_block:
                    cleaned_lines.extend(block)
                block = [line]
                is_binary_block = False
            elif 'Binary files' in line:
                is_binary_block = True
                block.append(line)
            else:
                block.append(line)

        if block and not is_binary_block:
            cleaned_lines.extend(block)
        return '\n'.join(cleaned_lines)

    git_patch = remove_binary_diffs(git_patch)
    test_result = {
        'git_patch': git_patch,
    }

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')

    # NOTE: this is NO LONGER the event stream, but an agent history that includes delegate agent's events
    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)

    # Save the output
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),  # SWE Bench specific
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


def filter_dataset(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if 'selected_ids' in data:
                selected_ids = data['selected_ids']
                logger.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                subset = dataset[dataset[filter_column].isin(selected_ids)]
                logger.info(f'Retained {subset.shape[0]} tasks after filtering')
                return subset
    skip_ids = [id for id in os.environ.get('SKIP_IDS', '').split(',') if id]
    if len(skip_ids) > 0:
        logger.info(f'Dataset size before filtering: {dataset.shape[0]} tasks')
        logger.info(f'Filtering {len(skip_ids)} tasks from "SKIP_IDS"...')
        logger.info(f'SKIP_IDS:\n{skip_ids}')
        filtered_dataset = dataset[~dataset[filter_column].isin(skip_ids)]
        logger.info(f'Dataset size after filtering: {filtered_dataset.shape[0]} tasks')
        return filtered_dataset
    return dataset


if __name__ == '__main__':
    # pdb.set_trace()
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--filter_dataset_after_sampling',
        action='store_true',
        help='if provided, filter dataset after sampling instead of before',
    )
    args, _ = parser.parse_known_args()

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    # dataset = load_dataset(args.dataset, split=args.split)
    # dataset = load_dataset(args.dataset)
    logger.info(f'Loading dataset {args.dataset} with split {args.split} ')
    dataset = load_dataset('json', data_files=args.dataset)
    dataset = dataset[args.split]
    swe_bench_tests = dataset.to_pandas()

    # Determine filter strategy based on flag
    filter_func = None
    if args.filter_dataset_after_sampling:
        # Pass filter as callback to apply after sampling
        def filter_func(df):
            return filter_dataset(df, 'instance_id')

        logger.info(
            f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks (filtering will occur after sampling)'
        )
    else:
        # Apply filter before sampling
        swe_bench_tests = filter_dataset(swe_bench_tests, 'instance_id')
        logger.info(
            f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks'
        )

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.log_completions = True
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    print(f'### OUTPUT FILE: {output_file} ###')
    instances = prepare_dataset(
        swe_bench_tests, output_file, args.eval_n_limit, filter_func=filter_func
    )

    if len(instances) > 0 and not isinstance(
        instances['FAIL_TO_PASS'][instances['FAIL_TO_PASS'].index[0]], str
    ):
        for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
            instances[col] = instances[col].apply(lambda x: str(x))
    # if LANGUAGE == "java": ##TODO:适配多语言的版本
    #     for col in ['issue_numbers', 'created_at']:
    #         instances[col] = instances[col].apply(lambda x: str(x))
    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=120 * 60,  # 2 hour PER instance should be more than enough
        max_retries=3,
    )
    # Check if any instances reached maximum retries
    check_maximum_retries_exceeded(metadata.eval_output_dir)
