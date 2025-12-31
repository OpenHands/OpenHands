import asyncio
import os

import pandas as pd  # type: ignore

# Assuming bfcl-eval is installed or we use a similar local structure
# The user mentioned: "Integrate bfcl-eval package for official metrics"
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
    instance_id = str(instance['id']).replace(
        '/', '_'
    )  # BFCL IDs might contain slashes

    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance_id}.')

    # Prepare instruction
    # BFCL usually has a question/prompt and associated functions
    question = instance['question']
    # We might need to format it with available tools?
    # For now, let's assume the agent can handle raw text or we format it.

    instruction = f'Question: {question}\n'
    # instruction += f"Available Functions: {instance['function']}\n"

    instruction += 'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
    instruction += AGENT_CLS_TO_INST_SUFFIX.get(metadata.agent_class, '')

    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

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

    last_agent_message = state.get_last_agent_message()
    model_answer_raw = last_agent_message.content if last_agent_message else ''

    output = EvalOutput(
        instance_id=instance_id,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result={
            'generated_text': model_answer_raw,
            # We will use bfcl-eval to score offline/post-hoc usually,
            # or we can try to score here if the package allows easy single-instance scoring.
        },
    )
    return output


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Path to the BFCL dataset (json/jsonl)',
    )
    args, _ = parser.parse_known_args()

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    llm_config.modify_params = False

    # Load dataset
    if args.dataset_path:
        if args.dataset_path.endswith('.json'):
            dataset_df = pd.read_json(args.dataset_path)
        elif args.dataset_path.endswith('.jsonl'):
            dataset_df = pd.read_json(args.dataset_path, lines=True)
        else:
            raise ValueError('Dataset must be .json or .jsonl')
    else:
        # Try to load from huggingface or default location?
        # For now require path or create dummy
        logger.warning('No dataset path provided, creating dummy dataset.')
        dataset_df = pd.DataFrame(
            [
                {
                    'id': 'test-0',
                    'question': 'What is the weather in San Francisco?',
                    'function': [
                        {
                            'name': 'get_weather',
                            'parameters': {'location': 'San Francisco'},
                        }
                    ],
                }
            ]
        )

    if 'instance_id' not in dataset_df.columns:
        if 'id' in dataset_df.columns:
            dataset_df['instance_id'] = dataset_df['id']
        else:
            dataset_df['instance_id'] = dataset_df.index.astype(str)

    metadata = make_metadata(
        llm_config=llm_config,
        dataset_name='bfcl',
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
