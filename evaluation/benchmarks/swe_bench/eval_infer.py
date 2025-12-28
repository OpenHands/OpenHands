import copy
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import pandas as pd
from tqdm import tqdm

from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.benchmarks.swe_bench.run_infer import get_instance_docker_image
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    get_default_sandbox_config_for_eval,
    get_openhands_config_for_eval,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.core.config import (
    LLMConfig,
    OpenHandsConfig,
    get_evaluation_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation
from openhands.utils.async_utils import call_async_from_sync

# TODO: migrate all swe-bench docker to ghcr.io/openhands
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')


def process_git_patch(patch):
    if not isinstance(patch, str):
        return ''

    if not patch.strip():
        # skip empty patches
        return ''

    patch = patch.replace('\r\n', '\n')
    # There might be some weird characters at the beginning of the patch
    # due to some OpenHands inference command outputs

    # FOR EXAMPLE:
    # git diff --no-color --cached 895f28f9cbed817c00ab68770433170d83132d90
    # [A[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[K0
    # diff --git a/django/db/models/sql/.backup.query.py b/django/db/models/sql/.backup.query.py
    # new file mode 100644
    # index 0000000000..fc13db5948

    # We "find" the first line that starts with "diff" and then we remove lines before it
    lines = patch.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            patch = '\n'.join(lines[i:])
            break

    patch = patch.rstrip() + '\n'  # Make sure the last line ends with a newline
    return patch


def get_config(metadata: EvalMetadata, instance: pd.Series) -> OpenHandsConfig:
    # We use a different instance image for the each instance of swe-bench eval
    base_container_image = get_instance_docker_image(instance['instance_id'])
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/OpenHands/OpenHands if you run into any issues.'
    )
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )
    config = get_openhands_config_for_eval(
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )
    return config


@dataclass
class ConditionalImports:
    """We instantiate the values in this dataclass differently if we're evaluating SWE-bench or SWE-Gym."""

    get_eval_report: Callable
    APPLY_PATCH_FAIL: str
    APPLY_PATCH_PASS: str


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    log_dir: str | None = None,
    runtime_failure_count: int = 0,
    conditional_imports: ConditionalImports | None = None,
) -> EvalOutput:
    """Evaluate agent performance on a SWE-bench problem instance.

    Note that this signature differs from the expected input to `run_evaluation`. Use
    `functools.partial` to provide optional arguments before passing to the evaluation harness.

    Args:
        log_dir (str | None, default=None): Path to directory where log files will be written. Must
        be provided if `reset_logger` is set.

        conditional_imports: A dataclass containing values that are imported differently based on
        whether we're evaluating SWE-bench or SWE-Gym.

    Raises:
        AssertionError: if the `reset_logger` flag is set without a provided log directory.

        AssertionError: if `conditional_imports` is not provided.
    """
    assert conditional_imports is not None, (
        'conditional_imports must be provided to run process_instance using multiprocessing'
    )

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        assert log_dir is not None, (
            "Can't reset logger without a provided log directory."
        )
        os.makedirs(log_dir, exist_ok=True)
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    config = get_config(metadata, instance)
    instance_id = instance.instance_id
    model_patch = instance['model_patch']
    test_spec = instance['test_spec']
    logger.info(f'Starting evaluation for instance {instance_id}.')

    if 'test_result' not in instance.keys():
        instance['test_result'] = {}
    instance['test_result']['report'] = {
        'empty_generation': False,
        'resolved': False,
        'failed_apply_patch': False,
        'error_eval': False,
        'test_timeout': False,
    }

    if model_patch == '':
        instance['test_result']['report']['empty_generation'] = True
        return EvalOutput(
            instance_id=instance_id,
            test_result=instance['test_result'],
            metadata=metadata,
        )

    # Increase resource_factor with increasing attempt_id
    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )
    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count
    metadata.details['remote_runtime_resource_factor'] = (
        config.sandbox.remote_runtime_resource_factor
    )

    try:
        runtime = create_runtime(config)
        call_async_from_sync(runtime.connect)
        # Get patch and save it to /tmp/patch.diff
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch file
            patch_file_path = os.path.join(temp_dir, 'patch.diff')
            with open(patch_file_path, 'w') as f:
                f.write(model_patch)
            runtime.copy_to(patch_file_path, '/tmp')
            # Eval script
            eval_script_path = os.path.join(temp_dir, 'eval.sh')
            with open(eval_script_path, 'w') as f:
                f.write(test_spec.eval_script)
            runtime.copy_to(eval_script_path, '/tmp')

        # Set +x
        action = CmdRunAction(command='chmod +x /tmp/eval.sh')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        # Apply patch
        exec_command = (
            'cd /testbed && '
            "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
            "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
            "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
            "echo 'APPLY_PATCH_FAIL')))"
        )
        action = CmdRunAction(command=exec_command)
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        assert isinstance(obs, CmdOutputObservation)
        apply_patch_output = obs.content
        assert isinstance(apply_patch_output, str)
        instance['test_result']['apply_patch_output'] = apply_patch_output

        if 'APPLY_PATCH_FAIL' in apply_patch_output:
            logger.info(
                f'[{instance_id}] {conditional_imports.APPLY_PATCH_FAIL}:\n{apply_patch_output}'
            )
            instance['test_result']['report']['failed_apply_patch'] = True

            return EvalOutput(
                instance_id=instance_id,
                test_result=instance['test_result'],
                metadata=metadata,
            )
        elif 'APPLY_PATCH_PASS' in apply_patch_output:
            logger.info(
                f'[{instance_id}] {conditional_imports.APPLY_PATCH_PASS}:\n{apply_patch_output}'
            )

            # Run eval script in background and save output to log file
            log_file = '/tmp/eval_output.log'
            action = CmdRunAction(command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!')
            action.set_hard_timeout(300)  # Short timeout just to get the process ID
            obs = runtime.run_action(action)

            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                pid = obs.content.split()[-1].strip()
                logger.info(
                    f'[{instance_id}] Evaluation process started with PID: {pid}'
                )

                # Poll for completion
                start_time = time.time()
                timeout = 1800  # 30 minutes
                while True:
                    seconds_elapsed = time.time() - start_time
                    if seconds_elapsed > timeout:
                        logger.info(
                            f'[{instance_id}] Evaluation timed out after {timeout} seconds'
                        )
                        instance['test_result']['report']['test_timeout'] = True
                        break
                    check_action = CmdRunAction(
                        command=f'ps -p {pid} > /dev/null; echo $?'
                    )
                    check_action.set_hard_timeout(300)
                    check_obs = runtime.run_action(check_action)
                    if (
                        isinstance(check_obs, CmdOutputObservation)
                        and check_obs.content.split()[-1].strip() == '1'
                    ):
                        logger.info(
                            f'[{instance_id}] Evaluation process completed after {seconds_elapsed} seconds'
                        )
                        break
                    logger.info(
                        f'[{instance_id}] [{seconds_elapsed:.0f}s] Evaluation still running, waiting...'
                    )
                    time.sleep(30)  # Wait for 30 seconds before checking again

                # Read the log file
                cat_action = CmdRunAction(command=f'cat {log_file}')
                cat_action.set_hard_timeout(300)
                cat_obs = runtime.run_action(cat_action)

                # Grade answer
                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    test_output = cat_obs.content
                    assert isinstance(test_output, str)
                    instance['test_result']['test_output'] = test_output

                    # Get report from test output
                    logger.info(f'[{instance_id}] Grading answer...')
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Create a directory structure that matches the expected format
                        # NOTE: this is a hack to make the eval report format consistent
                        # with the original SWE-Bench eval script
                        log_dir = os.path.join(temp_dir, 'logs', instance_id.lower())
                        os.makedirs(log_dir, exist_ok=True)
                        test_output_path = os.path.join(log_dir, 'test_output.txt')
                        with open(test_output_path, 'w') as f:
                            f.write(test_output)
                        try:
                            extra_kwargs = {}
                            if 'SWE-Gym' in metadata.dataset:
                                # SWE-Gym uses a different version of the package, hence a different eval report argument
                                extra_kwargs['log_path'] = test_output_path
                            else:
                                extra_kwargs['test_log_path'] = test_output_path
                            _report = conditional_imports.get_eval_report(
                                test_spec=test_spec,
                                prediction={
                                    'model_patch': model_patch,
                                    'instance_id': instance_id,
                                },
                                include_tests_status=True,
                                **extra_kwargs,
                            )
                            report = _report[instance_id]
                            logger.info(
                                f'[{instance_id}] report: {report}\nResult for {instance_id}: resolved: {report["resolved"]}'
                            )
                            instance['test_result']['report']['resolved'] = report[
                                'resolved'
                            ]
                        except Exception as e:
                            logger.error(
                                f'[{instance_id}] Error when getting eval report: {e}'
                            )
                            instance['test_result']['report']['resolved'] = False
                            instance['test_result']['report']['error_eval'] = True
            else:
                logger.info(f'[{instance_id}] Error when starting eval:\n{obs.content}')
                instance['test_result']['report']['error_eval'] = True

            return EvalOutput(
                instance_id=instance_id,
                test_result=instance['test_result'],
                metadata=metadata,
            )
        else:
            logger.info(
                f'[{instance_id}] Unexpected output when applying patch:\n{apply_patch_output}'
            )
            raise RuntimeError(
                instance_id,
                f'Unexpected output when applying patch:\n{apply_patch_output}',
                logger,
            )
    finally:
        runtime.close()


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--input-file',
        type=str,
        help='Path to input predictions file',
        required=True,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    args, _ = parser.parse_known_args()

    if 'SWE-Gym' in args.dataset:
        from swegym.harness.grading import get_eval_report
        from swegym.harness.run_evaluation import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
        )
        from swegym.harness.test_spec import (
            SWEbenchInstance,
            make_test_spec,
        )
        from swegym.harness.utils import load_swebench_dataset
    else:  # Newer version of SWE-Bench have different import paths
        from swebench.harness.grading import get_eval_report
        from swebench.harness.run_evaluation import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
        )
        from swebench.harness.test_spec.test_spec import (
            SWEbenchInstance,
            make_test_spec,
        )
        from swebench.harness.utils import load_swebench_dataset

    # Load SWE-Bench dataset
    full_dataset: list[SWEbenchInstance] = load_swebench_dataset(
        args.dataset, args.split
    )
    instance_id_to_instance = {
        instance['instance_id']: instance for instance in full_dataset
    }
    logger.info(
        f'Loaded dataset {args.dataset} with split {args.split} to run inference on.'
    )

    # Load predictions
    assert args.input_file.endswith('.jsonl'), 'Input file must be a jsonl file.'
    required_fields = ['instance_id', 'model_patch', 'test_result']
    with open(args.input_file) as f:
        predictions = pd.DataFrame.from_records(
            [
                {k: v for k, v in json.loads(line).items() if k in required_fields}
                for line in tqdm(f, desc='Loading predictions')
            ]
        )
    assert 'instance_id' in predictions.columns, (
        'Input file must contain instance_id column.'
    )

    if 'model_patch' not in predictions.columns and (
        'test_result' in predictions.columns
        and 'model_patch' in predictions['test_result'].iloc[0]
    ):
        raise ValueError(
            'Input file must contain model_patch column OR test_result column with model_patch field.'
        )
    assert len(predictions['instance_id'].unique()) == len(predictions), (
        'instance_id column must be unique.'
    )

    if 'model_patch' not in predictions.columns:
        predictions['model_patch'] = predictions['test_result'].apply(
            lambda x: x.get('git_patch', '')
        )
    assert {'instance_id', 'model_patch'}.issubset(set(predictions.columns)), (
        'Input file must contain instance_id and model_patch columns.'
    )

    # Process model_patch
    predictions['model_patch'] = predictions['model_patch'].apply(process_git_patch)

    # Merge predictions with dataset
    predictions['instance'] = predictions['instance_id'].apply(
        lambda x: instance_id_to_instance[x]
    )
    predictions['test_spec'] = predictions['instance'].apply(make_test_spec)

    # Prepare dataset
    output_file = args.input_file.replace('.jsonl', '.swebench_eval.jsonl')
    instances = prepare_dataset(predictions, output_file, args.eval_n_limit)

    # If possible, load the relevant metadata to avoid issues with `run_evaluation`.
    metadata: EvalMetadata | None = None
    metadata_filepath = os.path.join(os.path.dirname(args.input_file), 'metadata.json')
    if os.path.exists(metadata_filepath):
        with open(metadata_filepath, 'r') as metadata_file:
            data = metadata_file.read()
            metadata = EvalMetadata.model_validate_json(data)
    else:
        # Initialize with a dummy metadata when file doesn't exist
        metadata = EvalMetadata(
            agent_class='dummy_agent',  # Placeholder agent class
            llm_config=LLMConfig(model='dummy_model'),  # Minimal LLM config
            max_iterations=1,  # Minimal iterations
            eval_output_dir=os.path.dirname(
                args.input_file
            ),  # Use input file dir as output dir
            start_time=time.strftime('%Y-%m-%d %H:%M:%S'),  # Current time
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            .decode('utf-8')
            .strip(),  # Current commit
            dataset=args.dataset,  # Dataset name from args
            details={},
        )

    # The evaluation harness constrains the signature of `process_instance_func` but we need to
    # pass extra information. Build a new function object to avoid issues with multiprocessing.
    process_instance_func = partial(
        process_instance,
        log_dir=output_file.replace('.jsonl', '.logs'),
        # We have to explicitly pass these imports to the process_instance function, otherwise
        # they won't be available in the multiprocessing context.
        conditional_imports=ConditionalImports(
            get_eval_report=get_eval_report,
            APPLY_PATCH_FAIL=APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS=APPLY_PATCH_PASS,
        ),
    )

    run_evaluation(
        instances,
        metadata=metadata,
        output_file=output_file,
        num_workers=args.eval_num_workers,
        process_instance_func=process_instance_func,
    )

    # Load evaluated predictions & print number of resolved predictions
    evaluated_predictions = pd.read_json(output_file, lines=True)
    fields = ['resolved', 'failed_apply_patch', 'error_eval', 'empty_generation']

    def count_report_field(row, field):
        return row['test_result']['report'][field]

    report = {}
    for field in fields:
        count = evaluated_predictions.apply(
            count_report_field, args=(field,), axis=1
        ).sum()
        report[field] = count
        logger.info(
            f'# {field}: {count} / {len(evaluated_predictions)}. ({count / len(evaluated_predictions):.2%})'
        )
