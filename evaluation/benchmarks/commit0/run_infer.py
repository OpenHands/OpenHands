import asyncio
import json
import os
from collections import Counter
from typing import Any

import pandas as pd
from commit0.harness.constants import SPLIT
from datasets import load_dataset

import openhands.agenthub
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
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
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.events.serialization.event import event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
    'CodeActCommit0Agent': codeact_user_response,
}


def _get_commit0_workspace_dir_name(instance: pd.Series) -> str:
    return instance['repo'].split('/')[1]


def get_instruction(instance: pd.Series, metadata: EvalMetadata):
    workspace_dir_name = _get_commit0_workspace_dir_name(instance)
    # Prepare instruction
    test_cmd = instance['test']['test_cmd']
    test_dir = instance['test']['test_dir']
    # Instruction based on Anthropic's official trajectory
    # https://github.com/eschluntz/swe-bench-experiments/tree/main/evaluation/verified/20241022_tools_claude-3-5-sonnet-updated/trajs
    instruction = (
        '<uploaded_files>\n'
        f'/workspace/{workspace_dir_name}\n'
        '</uploaded_files>\n'
        f"I've uploaded a python code repository in the directory {workspace_dir_name}. Here is your task:\n\n"
        'Here is your task:\n\n'
        '  You need to complete the implementations for all functions (i.e., those with pass\n'
        '  statements) and pass the unit tests.\n\n'
        '  Do not change the names of existing functions or classes, as they may be referenced\n'
        '  from other code like unit tests, etc.\n\n'
        '  When you generate code, you must maintain the original formatting of the function\n'
        '  stubs (such as whitespaces), otherwise we will not able to search/replace blocks\n'
        '  for code modifications, and therefore you will receive a score of 0 for your generated\n'
        '  code.'
        '\n\n'
        'Here is the command to run the unit tests:\n'
        '<test_command>\n'
        f'{test_cmd} {test_dir}\n'
        '</test_command>\n\n'
        'Make a local git commit for each agent step for all code changes. If there is not change in current step, do not make a commit.'
    )

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
        )
    return instruction


# TODO: migrate all swe-bench docker to ghcr.io/openhands
DOCKER_IMAGE_PREFIX = os.environ.get(
    'EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/wentingzhao/'
)
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')


def get_instance_docker_image(repo_name: str) -> str:
    return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + repo_name).lower() + ':v0'


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    repo_name = instance['repo'].split('/')[1]
    base_container_image = get_instance_docker_image(repo_name)
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/OpenHands/OpenHands if you run into any issues.'
    )

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image

    config = get_openhands_config_for_eval(
        metadata=metadata,
        sandbox_config=sandbox_config,
        runtime=os.environ.get('RUNTIME', 'docker'),
        enable_browser=RUN_WITH_BROWSING,
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
    workspace_dir_name = _get_commit0_workspace_dir_name(instance)
    obs: CmdOutputObservation

    action = CmdRunAction(
        command=f'git clone -b commit0_combined https://github.com/{instance["repo"]}.git'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to git clone -b commit0_combined https://github.com/{instance["repo"]}.git: {str(obs)}',
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

    action = CmdRunAction(command='git checkout -b openhands')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0, f'Failed to git checkout new branch openhands: {str(obs)}'
    )

    # Install commit0
    action = CmdRunAction(command='/root/.cargo/bin/uv pip install commit0')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to install commit0: {str(obs)}',
    )
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
    workspace_dir_name = _get_commit0_workspace_dir_name(instance)

    action = CmdRunAction(command='git add .')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git add -A: {str(obs)}',
    )

    action = CmdRunAction(command='git commit -m "openhands edits"')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation)
        and (obs.exit_code == 0 or obs.exit_code == 1),
        f'Failed to git commit -m "openhands": {str(obs)}',
    )

    # Generate diff patch compared to base commit, excluding spec.pdf.bz2 files
    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f"git diff {instance['base_commit']} HEAD -- . ':(exclude)spec.pdf.bz2'"
        )
        action.set_hard_timeout(600 + 100 * n_retries)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                git_patch = obs.content.strip()
                break
            else:
                logger.info('Failed to get git diff, retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
        else:
            assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')

    test_dir = instance['test']['test_dir']
    action = CmdRunAction(
        command=f'{instance["test"]["test_cmd"]} --json-report --json-report-file=report.json --continue-on-collection-errors {test_dir} > test_output.txt 2>&1'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation),
        f'Failed to run test command: {str(obs)}',
    )
    # Read test output
    action = CmdRunAction(command='cat test_output.txt')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation),
        f'Failed to read test output: {str(obs)}',
    )
    test_output = obs.content.strip()
    # logger.info(f'Test output: {test_output}')

    # Save pytest exit code
    action = CmdRunAction(command='echo $?')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to save pytest exit code: {str(obs)}',
    )
    pytest_exit_code = obs.content.strip()
    # logger.info(f'Pytest exit code: {pytest_exit_code}')

    # Get test IDs from instance
    repo_name = instance['repo'].split('/')[1]
    repo_name = repo_name.replace('.', '-')
    action = CmdRunAction(command=f'commit0 get-tests {repo_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    test_ids = obs.content.strip().split('\n')

    # Read the test report
    action = CmdRunAction(command='cat report.json')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation),
        f'Failed to read test report: {str(obs)}',
    )
    json_report = obs.content.strip()

    try:
        report = json.loads(json_report)
        tests = {x['nodeid']: x['call'] for x in report['tests'] if 'call' in x}

        # Calculate test statistics
        status = []
        runtimes = []
        no_runs = 0

        for test_id in test_ids:
            if test_id in tests and tests[test_id] is not None:
                status.append(tests[test_id]['outcome'])
                runtimes.append(tests[test_id]['duration'])
                no_runs += 1
            else:
                status.append('failed')
                runtimes.append(0)

        status_counts = Counter(status)
        total_runtime = sum(runtimes) if no_runs > 0 else 0
        num_passed = status_counts.get('passed', 0) + status_counts.get('xfail', 0)
        passed_ratio = num_passed / len(status) if status else 0

        eval_result = {
            'name': workspace_dir_name,
            'sum': total_runtime,
            'passed': passed_ratio,
            'num_passed': num_passed,
            'num_tests': len(test_ids),
        }

    except json.JSONDecodeError:
        logger.error('Failed to parse test report JSON')
        eval_result = {
            'name': workspace_dir_name,
            'sum': 0,
            'passed': 0,
            'num_passed': 0,
            'num_tests': len(test_ids),
        }

    # Create tarball of workspace
    temp_zip = runtime.copy_from(f'/workspace/{workspace_dir_name}')

    commit0_dir = os.path.dirname(__file__)
    persistent_zip = os.path.join(commit0_dir, f'{workspace_dir_name}.zip')
    with open(temp_zip, 'rb') as src, open(persistent_zip, 'wb') as dst:
        dst.write(src.read())
    zip_file = persistent_zip
    return {
        'eval_result': eval_result,
        'git_patch': git_patch,
        'test_output': test_output,
        'pytest_exit_code': pytest_exit_code,
        'zip_file': zip_file,
    }


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(instance, metadata)
    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

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
        if (
            state.last_error
            and 'fatal error during agent execution' in state.last_error
            and 'stuck in a loop' not in state.last_error
        ):
            raise EvalException('Fatal error detected: ' + state.last_error)

        # ======= THIS IS Commit0 specific =======
        # Get git patch
        return_val = complete_runtime(runtime, instance)
        eval_result = return_val['eval_result']
        git_patch = return_val['git_patch']
        test_output = return_val['test_output']
        pytest_exit_code = return_val['pytest_exit_code']
        zip_file = return_val['zip_file']

        repo_name = instance['repo'].split('/')[1]
        zip_dest = os.path.join(
            metadata.eval_output_dir, 'repos', repo_name, f'{repo_name}.zip'
        )
        patch_file = os.path.join(
            metadata.eval_output_dir, 'repos', repo_name, f'{repo_name}_patch.diff'
        )
        test_output_file = os.path.join(
            metadata.eval_output_dir, 'repos', repo_name, f'{repo_name}_test_output.txt'
        )
        pytest_exit_code_file = os.path.join(
            metadata.eval_output_dir,
            'repos',
            repo_name,
            f'{repo_name}_pytest_exit_code.txt',
        )

        os.makedirs(os.path.dirname(zip_dest), exist_ok=True)
        os.rename(zip_file, zip_dest)

        write_targets = [
            (patch_file, git_patch),
            (test_output_file, test_output),
            (pytest_exit_code_file, pytest_exit_code),
        ]

        for write_target in write_targets:
            with open(write_target[0], 'w') as f:
                f.write(write_target[1])

        logger.info(
            f'Got evaluation result for repo {instance.instance_id}:\n--------\n{eval_result}\n--------'
        )
    finally:
        runtime.close()
    # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    test_result = {
        'eval_result': eval_result,
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
        instance=instance.to_dict(),
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


def commit0_setup(dataset: pd.DataFrame, repo_split: str) -> pd.DataFrame:
    """Setup Commit0 dataset based on split type.

    Args:
        dataset: Full Commit0 dataset
        repo_split: Split type ('all', 'lite' or specific repo name)

    Returns:
        Filtered dataset based on split type
    """
    filtered_dataset = pd.concat(
        [
            dataset[dataset['repo'].str.split('/').str[1] == repo]
            for repo in SPLIT.get(repo_split, [])
        ]
    )

    # Drop setup column if it exists
    if 'setup' in filtered_dataset.columns:
        filtered_dataset = filtered_dataset.drop('setup', axis=1)

    # Replace all forward slashes in instance_id with hyphens
    filtered_dataset['instance_id'] = filtered_dataset['repo'].str.split('/').str[1]

    return filtered_dataset


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='wentingzhao/commit0_combined',
        help='dataset to evaluate on, only test split exists for this HF dataset',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='this is the HF dataset split',
    )
    parser.add_argument(
        '--repo-split',
        type=str,
        default='lite',
        help='all, lite, or each repo name',
    )
    args, _ = parser.parse_known_args()

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    dataset = load_dataset(args.dataset, split=args.split)

    commit0_datasets = commit0_setup(dataset.to_pandas(), args.repo_split)

    logger.info(f'Loaded dataset {args.dataset} with reposplit {args.repo_split}')

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False
        llm_config.log_completions = True

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.repo_split.replace('/', '__')
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

    instances = prepare_dataset(commit0_datasets, output_file, args.eval_n_limit)

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=120 * 60,  # 2 hour PER instance should be more than enough
    )
