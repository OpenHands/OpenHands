import json
import logging
import multiprocessing as mp
import os
import pathlib
import signal
import subprocess
import time
import traceback
from contextlib import contextmanager
from inspect import signature
from typing import Any, Awaitable, Callable, Optional, TextIO

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from openhands.controller.state.state import State
from openhands.core.config import LLMConfig, SandboxConfig
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.condenser_config import (
    CondenserConfig,
    NoOpCondenserConfig,
)
from openhands.core.exceptions import (
    AgentRuntimeBuildError,
    AgentRuntimeDisconnectedError,
    AgentRuntimeError,
    AgentRuntimeNotFoundError,
    AgentRuntimeNotReadyError,
    AgentRuntimeTimeoutError,
    AgentRuntimeUnavailableError,
)
from openhands.core.logger import get_console_handler
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action
from openhands.events.action.message import MessageAction
from openhands.events.event import Event
from openhands.events.serialization.event import event_to_dict
from openhands.events.utils import get_pairs_from_events
from openhands.memory.condenser import get_condensation_metadata


class EvalMetadata(BaseModel):
    agent_class: str
    llm_config: LLMConfig
    agent_config: AgentConfig | None = None
    max_iterations: int
    eval_output_dir: str
    start_time: str
    git_commit: str
    dataset: str | None = None
    data_split: str | None = None
    details: dict[str, Any] | None = None
    condenser_config: CondenserConfig | None = None
    instruction_template_name: str | None = None


class EvalOutput(BaseModel):
    # NOTE: User-specified
    instance_id: str
    # output of the evaluation
    # store anything that is needed for the score calculation
    test_result: dict[str, Any]

    instruction: str | None = None

    # Interaction info
    metadata: EvalMetadata | None = None
    # list[tuple[dict[str, Any], dict[str, Any]]] - for compatibility with the old format
    history: (
        list[dict[str, Any]] | list[tuple[dict[str, Any], dict[str, Any]]] | None
    ) = None
    metrics: dict[str, Any] | None = None
    error: str | None = None

    # Optionally save the input test instance
    instance: dict[str, Any] | None = None


class EvalException(Exception):
    pass


class EvalTimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    def timeout_handler(signum, frame):
        raise EvalTimeoutException(f'Function timed out after {seconds} seconds')

    # Set up the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def codeact_user_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action], str] | None = None,
) -> str:
    encaps_str = (
        (
            'Your final answer MUST be encapsulated within <solution> and </solution>.\n'
            'For example: The answer to the question is <solution> 42 </solution>.\n'
        )
        if encapsulate_solution
        else ''
    )
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        'When you think you have solved the question, please use the finish tool and include your final answer in the message parameter of the finish tool.\n'
        f'{encaps_str}'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
    )

    if state.history:
        # check if the last action has an answer, if so, early exit
        if try_parse is not None:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            ans = try_parse(last_action)
            if ans is not None:
                return '/exit'

        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) >= 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
            )
    return msg


def cleanup():
    print('Cleaning up child processes...')
    for process in mp.active_children():
        print(f'Terminating child process: {process.name}')
        process.terminate()
        process.join()


def make_metadata(
    llm_config: LLMConfig,
    dataset_name: str,
    agent_class: str,
    max_iterations: int,
    eval_note: str | None,
    eval_output_dir: str,
    data_split: str | None = None,
    details: dict[str, Any] | None = None,
    agent_config: AgentConfig | None = None,
    condenser_config: CondenserConfig | None = None,
) -> EvalMetadata:
    model_name = llm_config.model.split('/')[-1]
    model_path = model_name.replace(':', '_').replace('@', '-')
    eval_note = f'_N_{eval_note}' if eval_note else ''

    eval_output_path = os.path.join(
        eval_output_dir,
        dataset_name,
        agent_class,
        f'{model_path}_maxiter_{max_iterations}{eval_note}',
    )

    pathlib.Path(eval_output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(eval_output_path, 'logs')).mkdir(
        parents=True, exist_ok=True
    )
    logger.info(f'Using evaluation output directory: {eval_output_path}')

    metadata = EvalMetadata(
        agent_class=agent_class,
        llm_config=llm_config,
        agent_config=agent_config,
        max_iterations=max_iterations,
        eval_output_dir=eval_output_path,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        .decode('utf-8')
        .strip(),
        dataset=dataset_name,
        data_split=data_split,
        details=details,
        condenser_config=condenser_config
        if condenser_config
        else NoOpCondenserConfig(),
        instruction_template_name=os.environ.get('INSTRUCTION_TEMPLATE_NAME'),
    )
    metadata_json = metadata.model_dump_json()
    logger.info(f'Metadata: {metadata_json}')
    with open(os.path.join(eval_output_path, 'metadata.json'), 'w') as f:
        f.write(metadata_json)

    return metadata


def prepare_dataset(
    dataset: pd.DataFrame,
    output_file: str,
    eval_n_limit: int,
    eval_ids: list[str] | None = None,
    skip_num: int | None = None,
    filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
):
    assert 'instance_id' in dataset.columns, (
        "Expected 'instance_id' column in the dataset. You should define your own unique identifier for each instance and use it as the 'instance_id' column."
    )
    id_column = 'instance_id'
    logger.info(f'Writing evaluation output to {output_file}')
    finished_ids: set[str] = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                finished_ids.add(str(data[id_column]))
        logger.warning(
            f'\nOutput file {output_file} already exists. Loaded {len(finished_ids)} finished instances.'
        )

    if eval_ids:
        eval_ids_converted = [dataset[id_column].dtype.type(id) for id in eval_ids]
        dataset = dataset[dataset[id_column].isin(eval_ids_converted)]
        logger.info(f'Limiting evaluation to {len(eval_ids)} specific instances.')
    elif skip_num and skip_num >= 0:
        skip_num = min(skip_num, len(dataset))
        dataset = dataset.iloc[skip_num:]
        logger.info(
            f'Starting evaluation with skipping first {skip_num} instances ({len(dataset)} instances to run).'
        )
        if eval_n_limit and eval_n_limit > 0:
            # Use fixed random seed 42 for sampling without replacement
            dataset = dataset.sample(
                min(eval_n_limit, len(dataset)), random_state=42, replace=False
            )
            logger.info(
                f'Randomly sampling {eval_n_limit} unique instances with random seed 42.'
            )
    elif eval_n_limit and eval_n_limit > 0:
        # Use fixed random seed 42 for sampling without replacement
        dataset = dataset.sample(
            min(eval_n_limit, len(dataset)), random_state=42, replace=False
        )
        logger.info(
            f'Randomly sampling {eval_n_limit} unique instances with random seed 42.'
        )

    if filter_func is not None:
        dataset = filter_func(dataset)
        logger.info(
            f'Applied filter after sampling: {len(dataset)} instances remaining'
        )

    def make_serializable(instance_dict: dict) -> dict:
        import numpy as np

        for k, v in instance_dict.items():
            if isinstance(v, np.ndarray):
                instance_dict[k] = v.tolist()
            elif isinstance(v, pd.Timestamp):
                instance_dict[k] = str(v)
            elif isinstance(v, dict):
                instance_dict[k] = make_serializable(v)
        return instance_dict

    new_dataset = [
        make_serializable(instance.to_dict())
        for _, instance in dataset.iterrows()
        if str(instance[id_column]) not in finished_ids
    ]
    logger.info(
        f'Finished instances: {len(finished_ids)}, Remaining instances: {len(new_dataset)}'
    )

    return pd.DataFrame(new_dataset)


def update_progress(
    result: EvalOutput,
    pbar: tqdm,
    output_fp: TextIO,
):
    """Update the progress bar and write the result to the output file."""
    pbar.update(1)
    pbar.set_description(f'Instance {result.instance_id}')
    pbar.set_postfix_str(f'Test Result: {str(result.test_result)[:300]}...')
    logger.info(
        f'Finished evaluation for instance {result.instance_id}: {str(result.test_result)[:300]}...\n'
    )
    output_fp.write(result.model_dump_json() + '\n')
    output_fp.flush()


def assert_and_raise(condition: bool, msg: str):
    """Raise an EvalException if the condition is not met.

    This will be used in conjunction with _process_instance_wrapper to handle retries. An EvalException should trigger a retry.
    """
    if not condition:
        raise EvalException(msg)


def log_skipped_maximum_retries_exceeded(instance, metadata, error, max_retries=5):
    """Log and skip the instance when maximum retries are exceeded.

    Args:
        instance: The instance that failed
        metadata: The evaluation metadata
        error: The error that occurred
        max_retries: The maximum number of retries that were attempted

    Returns:
        EvalOutput with the error information
    """
    from openhands.core.logger import openhands_logger as logger

    # Log the error
    logger.exception(error)
    logger.error(
        f'Maximum error retries reached for instance {instance.instance_id}. '
        f'Check maximum_retries_exceeded.jsonl, fix the issue and run evaluation again. '
        f'Skipping this instance and continuing with others.'
    )

    # Add the instance name to maximum_retries_exceeded.jsonl in the same folder as output.jsonl
    if metadata and metadata.eval_output_dir:
        retries_file_path = os.path.join(
            metadata.eval_output_dir,
            'maximum_retries_exceeded.jsonl',
        )
        try:
            # Write the instance info as a JSON line
            with open(retries_file_path, 'a') as f:
                import json

                # No need to get Docker image as we're not including it in the error entry

                error_entry = {
                    'instance_id': instance.instance_id,
                    'error': str(error),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                f.write(json.dumps(error_entry) + '\n')
            logger.info(f'Added instance {instance.instance_id} to {retries_file_path}')
        except Exception as write_error:
            logger.error(
                f'Failed to write to maximum_retries_exceeded.jsonl: {write_error}'
            )

    return EvalOutput(
        instance_id=instance.instance_id,
        test_result={},
        error=f'Maximum retries ({max_retries}) reached: {str(error)}',
        status='error',
    )


def check_maximum_retries_exceeded(eval_output_dir):
    """Check if maximum_retries_exceeded.jsonl exists and output a message."""
    from openhands.core.logger import openhands_logger as logger

    retries_file_path = os.path.join(eval_output_dir, 'maximum_retries_exceeded.jsonl')
    if os.path.exists(retries_file_path):
        logger.info(
            'ATTENTION: Some instances reached maximum error retries and were skipped.'
        )
        logger.info(f'These instances are listed in: {retries_file_path}')
        logger.info(
            'Fix these instances and run evaluation again with EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=false'
        )


def _process_instance_wrapper(
    process_instance_func: Callable[[pd.Series, EvalMetadata, bool], EvalOutput],
    instance: pd.Series,
    metadata: EvalMetadata,
    use_mp: bool,
    max_retries: int = 5,
    timeout_seconds: int | None = None,
) -> EvalOutput:
    """Wrap the process_instance_func to handle retries and errors."""
    runtime_failure_count = 0
    for attempt in range(max_retries + 1):
        try:
            kwargs = {}
            # check if process_instance_func accepts timeout_seconds parameter
            sig = signature(process_instance_func)
            if 'runtime_failure_count' in sig.parameters:
                kwargs['runtime_failure_count'] = runtime_failure_count

            if timeout_seconds is not None:
                with timeout(timeout_seconds):
                    result = process_instance_func(instance, metadata, use_mp, **kwargs)
            else:
                result = process_instance_func(instance, metadata, use_mp, **kwargs)
            return result
        except EvalTimeoutException as e:
            error = f'Timeout after {timeout_seconds} seconds'
            stacktrace = traceback.format_exc()
            msg = (
                '-' * 10
                + '\n'
                + f'Timeout ({timeout_seconds} seconds) in instance [{instance.instance_id}], Stopped evaluation for this instance.'
                + '\n'
                + '-' * 10
            )
            logger.exception(e)
            return EvalOutput(
                instance_id=instance.instance_id,
                test_result={},
                error=error,
            )
        except Exception as e:
            error = str(e)
            stacktrace = traceback.format_exc()
            if attempt == max_retries:
                msg = (
                    '-' * 10
                    + '\n'
                    + f'Error in instance [{instance.instance_id}]: {error}. Stacktrace:\n{stacktrace}'
                    + '\n'
                    + f'[Encountered after {max_retries} retries. Please check the logs and report the issue.]'
                    + '-' * 10
                )

                # Check if EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED is set to true
                skip_errors = (
                    os.environ.get(
                        'EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED', 'false'
                    ).lower()
                    == 'true'
                )

                if skip_errors:
                    # Use the dedicated function to log and skip maximum retries exceeded
                    return log_skipped_maximum_retries_exceeded(
                        instance, metadata, e, max_retries
                    )
                else:
                    # Raise an error after all retries & stop the evaluation
                    logger.exception(e)
                    raise RuntimeError(
                        f'Maximum error retries reached for instance {instance.instance_id}'
                    ) from e
            msg = (
                '-' * 10
                + '\n'
                + f'Error in instance [{instance.instance_id}]: {error}. Stacktrace:\n{stacktrace}'
                + '\n'
                + '-' * 10
                + f'[The above error occurred. Retrying... (attempt {attempt + 1} of {max_retries})]'
                + '-' * 10
                + '\n'
            )
            # e is likely an EvalException, so we can't directly infer it from type
            # but rather check if it's a fatal error
            # But it can also be AgentRuntime**Error (e.g., swe_bench/eval_infer.py)
            _error_str = type(e).__name__ + ': ' + str(e)
            if is_fatal_runtime_error(_error_str):
                runtime_failure_count += 1
                msg += f'Runtime disconnected error detected for instance {instance.instance_id}, runtime failure count: {runtime_failure_count}'
                msg += '\n' + '-' * 10 + '\n'
            logger.error(msg)
            time.sleep(5)


def _process_instance_wrapper_mp(args):
    """Wrapper for multiprocessing, especially for imap_unordered."""
    return _process_instance_wrapper(*args)


def run_evaluation(
    dataset: pd.DataFrame,
    metadata: EvalMetadata | None,
    output_file: str,
    num_workers: int,
    process_instance_func: Callable[
        [pd.Series, EvalMetadata, bool], Awaitable[EvalOutput]
    ],
    max_retries: int = 5,  # number of retries for each instance
    timeout_seconds: int | None = None,
):
    use_multiprocessing = num_workers > 1

    if metadata is not None:
        logger.info(
            f'Evaluation started with Agent {metadata.agent_class}:\n'
            f'model {metadata.llm_config.model}, max iterations {metadata.max_iterations}.\n'
        )
    else:
        logger.warning('Running evaluation without metadata.')
        logger.info(f'Evaluation started with {num_workers} workers.')

    total_instances = len(dataset)
    pbar = tqdm(total=total_instances, desc='Instances processed')
    output_fp = open(output_file, 'a')

    try:
        if use_multiprocessing:
            with mp.Pool(num_workers) as pool:
                args_iter = (
                    (
                        process_instance_func,
                        instance,
                        metadata,
                        True,
                        max_retries,
                        timeout_seconds,
                    )
                    for _, instance in dataset.iterrows()
                )
                results = pool.imap_unordered(_process_instance_wrapper_mp, args_iter)
                for result in results:
                    update_progress(result, pbar, output_fp)
        else:
            for _, instance in dataset.iterrows():
                result = _process_instance_wrapper(
                    process_instance_func=process_instance_func,
                    instance=instance,
                    metadata=metadata,
                    use_mp=False,
                    max_retries=max_retries,
                )
                update_progress(result, pbar, output_fp)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt received. Cleaning up...\n')
        cleanup()

    output_fp.close()
    logger.info('\nEvaluation finished.\n')

    # Check if any instances reached maximum retries
    if metadata and metadata.eval_output_dir:
        check_maximum_retries_exceeded(metadata.eval_output_dir)


def reset_logger_for_multiprocessing(
    logger: logging.Logger, instance_id: str, log_dir: str
):
    """Reset the logger for multiprocessing.

    Save logs to a separate file for each process, instead of trying to write to the
    same file/console from multiple processes.
    """
    # Set up logger
    log_file = os.path.join(
        log_dir,
        f'instance_{instance_id}.log',
    )
    # Remove all existing handlers from logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # add console handler to print ONE line
    console_handler = get_console_handler(log_level=logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            f'Instance {instance_id} - ' + '%(asctime)s - %(levelname)s - %(message)s'
        )
    )
    logger.addHandler(console_handler)
    logger.info(
        f'Starting evaluation for instance {instance_id}.\n'
        f'Hint: run "tail -f {log_file}" to see live logs in a separate shell'
    )
    # Only log WARNING or higher to console
    console_handler.setLevel(logging.WARNING)

    # Log INFO and above to file
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def update_llm_config_for_completions_logging(
    llm_config: LLMConfig,
    eval_output_dir: str,
    instance_id: str,
) -> LLMConfig:
    """Update the LLM config for logging completions."""
    if llm_config.log_completions:
        llm_config.log_completions_folder = os.path.join(
            eval_output_dir, 'llm_completions', instance_id
        )
        logger.info(
            f'Logging LLM completions for instance {instance_id} to '
            f'{llm_config.log_completions_folder}'
        )
    return llm_config


# history is now available as a filtered stream of events, rather than list of pairs of (Action, Observation)
# we rebuild the pairs here
# for compatibility with the existing output format in evaluations
# remove this when it's no longer necessary
def compatibility_for_eval_history_pairs(
    history: list[Event],
) -> list[tuple[dict, dict]]:
    history_pairs = []

    for action, observation in get_pairs_from_events(history):
        history_pairs.append((event_to_dict(action), event_to_dict(observation)))

    return history_pairs


def is_fatal_evaluation_error(error: str | None) -> bool:
    """The AgentController class overrides last error for certain exceptions
    We want to ensure those exeption do not overlap with fatal exceptions defined here
    This is because we do a comparisino against the stringified error
    """
    if not error:
        return False

    FATAL_EXCEPTIONS = [
        AgentRuntimeError,
        AgentRuntimeBuildError,
        AgentRuntimeTimeoutError,
        AgentRuntimeUnavailableError,
        AgentRuntimeNotReadyError,
        AgentRuntimeDisconnectedError,
        AgentRuntimeNotFoundError,
        ConnectionError,
    ]

    if any(exception.__name__ in error for exception in FATAL_EXCEPTIONS):
        logger.error(f'Fatal evaluation error detected: {error}')
        return True

    return False


def is_fatal_runtime_error(error: str | None) -> bool:
    if not error:
        return False

    FATAL_RUNTIME_ERRORS = [
        AgentRuntimeTimeoutError,
        AgentRuntimeUnavailableError,
        AgentRuntimeDisconnectedError,
        AgentRuntimeNotFoundError,
    ]

    if any(exception.__name__ in error for exception in FATAL_RUNTIME_ERRORS):
        logger.error(f'Fatal runtime error detected: {error}')
        return True

    return False


def get_metrics(state: State) -> dict[str, Any]:
    """Extract metrics for evaluations.

    Prefer ConversationStats (source of truth) and fall back to state.metrics for
    backward compatibility.
    """
    metrics: dict[str, Any]
    try:
        if getattr(state, 'conversation_stats', None):
            combined = state.conversation_stats.get_combined_metrics()
            metrics = combined.get()
        elif getattr(state, 'metrics', None):
            metrics = state.metrics.get()
        else:
            metrics = {}
    except Exception:
        metrics = state.metrics.get() if getattr(state, 'metrics', None) else {}

    metrics['condenser'] = get_condensation_metadata(state)
    return metrics


def get_default_sandbox_config_for_eval() -> SandboxConfig:
    return SandboxConfig(
        use_host_network=False,
        # large enough timeout, since some testcases take very long to run
        timeout=300,
        api_key=os.environ.get('ALLHANDS_API_KEY', None),
        runtime_startup_env_vars={'NO_CHANGE_TIMEOUT_SECONDS': '30'},
        remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
        keep_runtime_alive=False,
        remote_runtime_init_timeout=3600,
        remote_runtime_api_timeout=120,
        remote_runtime_enable_retries=True,
        remote_runtime_class='sysbox',
    )


def get_openhands_config_for_eval(
    metadata: EvalMetadata | None = None,
    sandbox_config: SandboxConfig | None = None,
    runtime: str | None = None,
    max_iterations: int | None = None,
    default_agent: str | None = None,
    enable_browser: bool = False,
    workspace_base: str | None = None,
    workspace_mount_path: str | None = None,
):
    """Create an OpenHandsConfig with common patterns used across evaluation scripts.

    This function provides a standardized way to create OpenHands configurations
    for evaluation runs, with sensible defaults that match the patterns used in
    most run_infer.py scripts. Individual evaluation scripts can override specific
    attributes as needed.

    Args:
        metadata: EvalMetadata containing agent class, max iterations, etc.
        sandbox_config: Custom sandbox config. If None, uses get_default_sandbox_config_for_eval()
        runtime: Runtime type. If None, uses environment RUNTIME or 'docker'
        max_iterations: Max iterations for the agent. If None, uses metadata.max_iterations
        default_agent: Agent class name. If None, uses metadata.agent_class
        enable_browser: Whether to enable browser functionality
        workspace_base: Workspace base path. Defaults to None
        workspace_mount_path: Workspace mount path. Defaults to None

    Returns:
        OpenHandsConfig: Configured for evaluation with eval-specific overrides applied
    """
    # Defer import to avoid circular imports at module load time
    from openhands.core.config.openhands_config import (
        OpenHandsConfig as _OHConfig,  # type: ignore
    )

    # Use provided sandbox config or get default
    if sandbox_config is None:
        sandbox_config = get_default_sandbox_config_for_eval()

    # Extract values from metadata if provided
    if metadata is not None:
        if max_iterations is None:
            max_iterations = metadata.max_iterations
        if default_agent is None:
            default_agent = metadata.agent_class

    # Use environment runtime or default
    if runtime is None:
        runtime = os.environ.get('RUNTIME', 'docker')

    # Provide sensible defaults if still None
    if default_agent is None:
        default_agent = 'CodeActAgent'
    if max_iterations is None:
        max_iterations = 50

    # Always use repo-local .eval_sessions directory (absolute path)
    eval_store = os.path.abspath(os.path.join(os.getcwd(), '.eval_sessions'))

    # Create the base config with evaluation-specific overrides
    config = _OHConfig(
        default_agent=default_agent,
        run_as_openhands=False,
        runtime=runtime,
        max_iterations=max_iterations,
        enable_browser=enable_browser,
        sandbox=sandbox_config,
        workspace_base=workspace_base,
        workspace_mount_path=workspace_mount_path,
        file_store='local',
        file_store_path=eval_store,
    )

    return config
