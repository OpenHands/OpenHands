import asyncio
import os
from typing import Any

import pandas as pd  # type: ignore

try:
    from tau_bench.agents.base import Agent as TauAgent  # type: ignore
    from tau_bench.envs import get_env  # type: ignore
    from tau_bench.types import EnvInfo  # type: ignore
except ImportError:
    TauAgent = Any
    get_env = Any
    EnvInfo = Any

from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    codeact_user_response,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    OpenHandsConfig,
    get_evaluation_parser,
    get_llm_config_arg,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import MessageAction
from openhands.utils.async_utils import call_async_from_sync

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you think you have completed the request, please finish the interaction using the "finish" tool.\n'
}


def get_config(
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = 'python:3.12-bookworm'
    config = get_openhands_config_for_eval(
        metadata=metadata,
        runtime='docker',
        sandbox_config=sandbox_config,
    )
    config.set_llm_config(metadata.llm_config)
    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False
    return config


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(metadata)
    instance_id = str(instance['instance_id'])

    # Setup the logger properly
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance_id}.')

    # Initialize Tau-Bench environment
    instance['env']
    instance['task_index']

    # Initialize runtime
    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    # Note: We need to figure out how to bridge Tau-Bench environment with OpenHands agent.
    # OpenHands agents expect to interact with a runtime (shell/browser).
    # Tau-Bench environments provide a python interface.
    # For now, we will assume we can run python code in the runtime to interact with Tau-Bench,
    # OR we adapt the agent to call Tau-Bench API.

    # Given OpenHands agents are general purpose, we probably want to expose Tau-Bench tools
    # as Python functions available in the runtime, or standard tools.

    # Let's inspect how Tau-Bench works. It seems it requires `tau-bench` package.
    # The user request mentioned: "Integrate sierra-research/tau-bench package for dataset and evaluation"

    # Since I don't have the package installed yet, I will write the skeleton and then install/mock it.

    instruction = instance['instruction']
    instruction += 'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
    instruction += AGENT_CLS_TO_INST_SUFFIX.get(metadata.agent_class, '')

    state: State | None = asyncio.run(
        run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
            fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                metadata.agent_class
            ),
        )
    )

    if state is None:
        raise ValueError('State should not be None.')

    metrics = get_metrics(state)
    histories = compatibility_for_eval_history_pairs(state.history)

    # Retrieve result from the state or runtime if possible
    # For Tau-Bench, we typically need to check if the goal was achieved in the env.

    # Placeholder for actual score calculation
    score = 0.0

    output = EvalOutput(
        instance_id=instance_id,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result={
            'score': score,
        },
    )
    return output


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--env',
        type=str,
        default='retail',
        help='Tau-Bench environment name (retail, airline)',
    )
    args, _ = parser.parse_known_args()

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    llm_config.modify_params = False

    # Load dataset
    # We need to load tasks from Tau-Bench
    # Since we can't import tau_bench yet, we might fail here.
    # But I will write the import and let the user/system install it.
    try:
        from tau_bench.envs import get_env  # type: ignore
    except ImportError:
        logger.error(
            'Tau-Bench not installed. Please install it via `pip install tau-bench`'
        )
        # For now, we create a dummy dataset to allow syntax checking
        dataset_df = pd.DataFrame(
            [
                {
                    'instance_id': '0',
                    'env': 'retail',
                    'task_index': 0,
                    'instruction': 'Test instruction',
                }
            ]
        )
    else:
        # Load tasks from the environment
        env = get_env(args.env)
        tasks = env.get_tasks()
        data = []
        for i, task in enumerate(tasks):
            data.append(
                {
                    'instance_id': f'{args.env}_{i}',
                    'env': args.env,
                    'task_index': i,
                    'instruction': task.instruction,
                    'ground_truth': task.actions,  # Or whatever ground truth it provides
                }
            )
        dataset_df = pd.DataFrame(data)

    metadata = make_metadata(
        llm_config=llm_config,
        dataset_name=f'tau-bench-{args.env}',
        agent_class=args.agent_cls,
        max_iterations=args.max_iterations,
        eval_note=args.eval_note,
        eval_output_dir=args.eval_output_dir,
        data_split=args.data_split,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    dataset = prepare_dataset(
        dataset_df, output_file=output_file, eval_n_limit=args.eval_n_limit
    )

    run_evaluation(
        dataset=dataset,
        metadata=metadata,
        output_file=output_file,
        num_workers=args.eval_num_workers,
        process_instance_func=process_instance,
    )
