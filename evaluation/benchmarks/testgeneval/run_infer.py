import asyncio
import json
import os
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import toml
from datasets import load_dataset

import openhands.agenthub
from evaluation.benchmarks.testgeneval.constants import MAP_REPO_VERSION_TO_SPECS
from evaluation.benchmarks.testgeneval.prompt import (
    CODEACT_TESTGEN_PROMPT,
    CODEACT_TESTGEN_PROMPT_ITERATE,
)
from evaluation.benchmarks.testgeneval.utils import get_test_directives
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    codeact_user_response,
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
    SandboxConfig,
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

RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}


def _preprocess_instance(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
    return d


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    return f'{instance.repo}__{instance.version}'.replace('/', '__')


def get_instruction(instance: pd.Series, metadata: EvalMetadata):
    # workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    # Prepare instruction
    coverage_command = ' '.join(
        [
            MAP_REPO_VERSION_TO_SPECS[instance['repo']][instance['version']][
                'test_cmd'
            ],
            *get_test_directives(instance),
        ]
    )

    # Testing general agents
    prompt_to_use = (
        CODEACT_TESTGEN_PROMPT_ITERATE
        if instance['full_pred'] is not None
        else CODEACT_TESTGEN_PROMPT
    )
    instruction = prompt_to_use.format(
        code_file=os.path.join('/testbed', instance.code_file),
        test_file=os.path.join('/testbed', instance.test_file),
        coverage_command=coverage_command,
        code_src=instance['code_src'],
        imports='\n'.join(instance.local_imports),
        workspace_dir_name=_get_swebench_workspace_dir_name(instance),
    )

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
        )

    return instruction


# TODO: migrate all swe-bench docker to ghcr.io/openhands
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/kdjain/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')


def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    # We use a different instance image for the each instance of TestGenEval
    base_container_image = get_instance_docker_image(instance['instance_id_swebench'])
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/OpenHands/OpenHands if you run into any issues.'
    )

    sandbox_config = SandboxConfig(
        base_container_image=base_container_image,
        enable_auto_lint=True,
        use_host_network=False,
        # large enough timeout, since some testcases take very long to run
        timeout=300,
        # Add platform to the sandbox config to solve issue 4401
        platform='linux/amd64',
        api_key=os.environ.get('ALLHANDS_API_KEY', None),
        remote_runtime_api_url=os.environ.get(
            'SANDBOX_REMOTE_RUNTIME_API_URL', 'http://localhost:8000'
        ),
        keep_runtime_alive=False,
        remote_runtime_init_timeout=3600,
    )

    config = get_openhands_config_for_eval(
        metadata=metadata,
        sandbox_config=sandbox_config,
        runtime=os.environ.get('RUNTIME', 'docker'),
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['id']
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

    instance['instance_id'] = instance['instance_id_swebench']

    # Set instance id
    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance['instance_id_swebench']}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc"""
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0, f'Failed to export SWE_INSTANCE_ID: {str(obs)}'
    )

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')

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
    swe_prediction = 'test_suite.py'
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the full path for the desired file name within the temporary directory
        temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
        # Write to the file with the desired name within the temporary directory
        with open(temp_file_path, 'w') as f:
            if not isinstance(instance, dict):
                preprocessed_instance = _preprocess_instance(instance.to_dict())
                json.dump([preprocessed_instance], f)
            else:
                preprocessed_instance = _preprocess_instance(instance)
                json.dump([preprocessed_instance], f)

        # Copy the file to the desired location
        runtime.copy_to(temp_file_path, '/swe_util/eval_data/instances/')

        if instance['full_pred'] is not None:
            temp_file_path_pred = os.path.join(temp_dir, swe_prediction)
            with open(temp_file_path_pred, 'w') as f:
                f.write(instance['full_pred'])

            runtime.copy_to(temp_file_path_pred, '/tmp')

            # Copy the file to the desired location
            action = CmdRunAction(
                command=f'cp /tmp/test_suite.py /testbed/{instance["test_file"]}'
            )
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            assert_and_raise(
                obs.exit_code == 0, f'Failed to copy test file: {str(obs)}'
            )

            action = CmdRunAction(
                command='git -C /testbed add . && git -C /testbed commit -m "Add test file"'
            )
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            assert_and_raise(obs.exit_code == 0, f'Failed to cat ~/.bashrc: {str(obs)}')

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
    try:
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
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
        )

        action = CmdRunAction(command=f'cat {instance.test_file}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to find file: {instance.test_file} in /workspace/{workspace_dir_name}',
        )

        test_suite = obs.content.strip()
    except Exception:
        # Print stack trace
        print('Skipping, exception in complete_runtime')
        print(traceback.format_exc())
        test_suite = instance['full_pred'] if instance['full_pred'] is not None else ''

    # action = CmdRunAction(command='git add -A')
    # action.set_hard_timeout(600)
    # logger.info(action, extra={'msg_type': 'ACTION'})
    # obs = runtime.run_action(action)
    # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    # assert_and_raise(obs.exit_code == 0, f'Failed to git add -A: {str(obs)}')

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {
        'test_suite': test_suite,
    }


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(instance, metadata)
    start_time = time.time()  # Track start time

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.id}.')

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
        return_val = complete_runtime(runtime, instance)
        test_suite = return_val['test_suite']
        logger.info(
            f'Got test suite for instance {instance.instance_id}:\n--------\n{test_suite}\n--------'
        )
    finally:
        runtime.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        f'Evaluation for instance {instance.instance_id} took {elapsed_time:.2f} seconds.'
    )

    # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    test_result = {
        'test_suite': test_suite,
        'elapsed_time': elapsed_time,
    }

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')

    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)

    # Save the output
    output = EvalOutput(
        instance_id=instance.id,
        instruction=instruction,
        instance=_preprocess_instance(instance.to_dict()),  # SWE Bench specific
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    # print(output)
    return output


def prepare_dataset_pre(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
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

                subset['instance_id_swebench'] = subset['instance_id']
                subset['instance_id'] = subset['id']
                return subset

    dataset['instance_id_swebench'] = dataset['instance_id']
    dataset['instance_id'] = dataset['id']
    return dataset


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='kjain/testgenevallite',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--testfile_start',
        action='store_true',
        help='Whether to start from the 0 shot test file',
    )

    parser.add_argument(
        '--zero_shot_path',
        type=str,
        help='Path to the zero shot test file predictions',
    )
    args, _ = parser.parse_known_args()

    if args.testfile_start and not args.zero_shot_path:
        raise ValueError(
            'If you want to start from the 0 shot test file, you must provide the path to the zero shot test file predictions'
        )

    preds_map = {}
    if args.testfile_start:
        with open(args.zero_shot_path, 'r') as f:
            for line in f:
                pred = json.loads(line)
                preds_map[pred['id']] = pred['preds']['full'][0]

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    dataset = load_dataset(args.dataset, split=args.split)
    logger.info(f'Loaded dataset {args.dataset} with split {args.split}')
    testgeneval_filepairs = prepare_dataset_pre(dataset.to_pandas(), 'id')

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.log_completions = True
        # modify_params must be False for evaluation purpose, for reproducibility and accuracy of results
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
    instances = prepare_dataset(testgeneval_filepairs, output_file, args.eval_n_limit)

    if not instances.empty:
        instances['full_pred'] = (
            instances['instance_id']
            .map(preds_map)
            .apply(lambda x: x if pd.notna(x) else None)
        )

        run_evaluation(
            instances, metadata, output_file, args.eval_num_workers, process_instance
        )
