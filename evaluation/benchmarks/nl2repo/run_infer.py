import asyncio
import copy
import json
import os
import shutil
from typing import Any
import tempfile
import pandas as pd

from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    get_openhands_config_for_eval,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    get_metrics,
    is_fatal_evaluation_error,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    get_evaluation_parser,
    get_llm_config_arg,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.events.serialization.event import event_to_dict
from openhands.runtime.base import Runtime
from openhands.storage.locations import get_conversation_dir
from openhands.utils.async_utils import call_async_from_sync

WORKING_DIR = "/workspace"
NL2REPO_BASE_IMAGE = os.environ.get(
    'NL2REPO_BASE_IMAGE', 
    'all-hands-ai/openhands:0.56-nl2repo' # self-hosted image
)

def get_instruction(instance: pd.Series, metadata: EvalMetadata) -> MessageAction:
    instruction = "According to the start.md in the workspace, implement the entire project as per the requirements specified in the document, ensuring that the final product can be directly run in the current directory. The running requirements should comply with the <API Usage Guide> section of the document. Please complete this task step by step."
    return MessageAction(content=instruction)

def get_instance_docker_image(instance_id: str) -> str:
    return NL2REPO_BASE_IMAGE

def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    base_container_image = get_instance_docker_image(instance['instance_id'])
    
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = True
    sandbox_config.platform = 'linux/amd64'

    config = get_openhands_config_for_eval(
        metadata=metadata,
        enable_browser=False,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )
    
    cur_llm_config = update_llm_config_for_completions_logging(
        metadata.llm_config, 
        metadata.eval_output_dir, 
        instance['instance_id']
    )
    config.set_llm_config(cur_llm_config)
    
    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=False,
        enable_llm_editor=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
    )
    config.set_agent_config(agent_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
    metadata: EvalMetadata,
):
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization for NL2Repo')
    logger.info('-' * 30)
        
    obs: CmdOutputObservation

    action = CmdRunAction(command=f'mkdir -p {WORKING_DIR} && cd {WORKING_DIR}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to create workspace: {str(obs)}')

    with tempfile.TemporaryDirectory() as temp_dir:
        description = instance['description']
        temp_file_path = os.path.join(temp_dir, 'start.md')
        with open(temp_file_path, 'w') as f:
            f.write(description)
        runtime.copy_to(temp_file_path, '/workspace')

    action = CmdRunAction(command='source ~/.bashrc')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    action = CmdRunAction(
        command=f'git init && git config user.email "agent@z.ai" && git config user.name "Zai Agent"'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    logger.info('-' * 30)
    logger.info('END Runtime Initialization for NL2Repo')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
    metadata: EvalMetadata,
) -> dict[str, Any]:
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion for NL2Repo')
    logger.info('-' * 30)
    
    instance_id = instance['instance_id']
    instance_output_dir = os.path.join(metadata.eval_output_dir, 'instances', instance_id)
    os.makedirs(instance_output_dir, exist_ok=True)
    
    obs: CmdOutputObservation
    
    action = CmdRunAction(command=f'cd {WORKING_DIR}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    
    action = CmdRunAction(command=f'find {WORKING_DIR} -type f -name "*.py"')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    
    generated_files = []
    if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
        generated_files = [f for f in obs.content.strip().split('\n') if f]
    
    action = CmdRunAction(command=f'find {WORKING_DIR} -type f | grep -v __pycache__ | grep -v ".git"')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    
    file_tree = []
    if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
        file_tree = [f for f in obs.content.strip().split('\n') if f]
    
    action = CmdRunAction(command=f'cat {WORKING_DIR}/pyproject.toml 2>/dev/null || echo "NO_PYPROJECT"')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    
    has_pyproject = False
    pyproject_content = ""
    if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
        if "NO_PYPROJECT" not in obs.content:
            has_pyproject = True
            pyproject_content = obs.content
    
    workspace_zip = runtime.copy_from(WORKING_DIR)
    workspace_dir = instance_output_dir
    shutil.unpack_archive(str(workspace_zip), workspace_dir)
    os.unlink(str(workspace_zip))
    logger.info(f'Successfully copied /workspace to {workspace_dir}')

    logger.info('-' * 30)
    logger.info('END Runtime Completion for NL2Repo')
    logger.info('-' * 30)

    return {
        'generated_files': generated_files,
        'file_tree': file_tree,
        'has_pyproject': has_pyproject,
        'pyproject_content': pyproject_content,
        'workspace_dir': workspace_dir,
    }


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    config = get_config(instance, metadata)

    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, '
            f'setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count
    metadata.details['remote_runtime_resource_factor'] = config.sandbox.remote_runtime_resource_factor

    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    try:
        initialize_runtime(runtime, instance, metadata)

        message_action = get_instruction(instance, metadata)

        state, status = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                fake_user_response_fn=codeact_user_response,
            )
        )

        truncated = status['truncated']

        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

        try:
            return_val = complete_runtime(runtime, instance, metadata)
            logger.info(
                f'Completed instance {instance.instance_id}: '
                f'{len(return_val.get("generated_files", []))} Python files generated'
            )
        except Exception as e:
            logger.error(f'Error completing runtime for instance {instance.instance_id}: {e}')
            return_val = {
                'generated_files': [],
                'file_tree': [],
                'has_pyproject': False,
                'error': str(e),
            }
            raise e
    finally:
        try:
            event_stream = runtime.event_stream
            sid = event_stream.sid
            file_store = event_stream.file_store
            session_dir = get_conversation_dir(sid, event_stream.user_id)
            file_store.delete(session_dir)
        except Exception as e:
            logger.warning(f'Failed to clean up session folder: {e}')
        runtime.close()

    test_result = {
        'generated_files': return_val.get('generated_files', []),
        'file_tree': return_val.get('file_tree', []),
        'has_pyproject': return_val.get('has_pyproject', False),
        'truncated': truncated,
        'workspace_dir': return_val.get('workspace_dir'),
    }

    if state is None:
        raise ValueError('State should not be None.')

    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)

    instruction = message_action.content
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


def load_nl2repo_dataset(data_path: str) -> pd.DataFrame:
    if data_path.endswith('.jsonl'):
        instances = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    instances.append(json.loads(line))
        return pd.DataFrame(instances)
    
    elif os.path.isdir(data_path):
        instances = []
        test_files_path = data_path if 'test_files' in data_path else os.path.join(data_path, 'test_files')
        
        if not os.path.exists(test_files_path):
            test_files_path = data_path
        
        for project_folder in os.listdir(test_files_path):
            project_path = os.path.join(test_files_path, project_folder)
            if not os.path.isdir(project_path):
                continue
            
            try:
                instance = {'instance_id': project_folder}
                
                txt_files = [f for f in os.listdir(project_path) if f.endswith('.txt')]
                if txt_files:
                    with open(os.path.join(project_path, txt_files[0]), 'r') as f:
                        instance['test_case_count'] = int(f.read().strip())
                
                cmd_files = [f for f in os.listdir(project_path) if f.endswith('.json') and 'commands' in f]
                if cmd_files:
                    with open(os.path.join(project_path, cmd_files[0]), 'r') as f:
                        instance['test_commands'] = json.load(f)
                
                files_files = [f for f in os.listdir(project_path) if f.endswith('.json') and 'files' in f]
                if files_files:
                    with open(os.path.join(project_path, files_files[0]), 'r') as f:
                        instance['test_files'] = json.load(f)
                
                md_files = [f for f in os.listdir(project_path) if f.endswith('.md')]
                if md_files:
                    with open(os.path.join(project_path, md_files[0]), 'r', encoding='utf-8') as f:
                        instance['problem_statement'] = f.read()
                
                instances.append(instance)
                logger.info(f'Loaded instance: {project_folder}')
                
            except Exception as e:
                logger.error(f'Error loading project {project_folder}: {e}')
        
        return pd.DataFrame(instances)


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--split', type=str, default='train')
    args, _ = parser.parse_known_args()

    nl2repo_tests = load_nl2repo_dataset(args.dataset)
    
    logger.info(f'Loaded NL2Repo dataset with {len(nl2repo_tests)} tasks')

    config = vars(args)
    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config, args.config_file)
        llm_config.log_completions = True
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    dataset_description = 'nl2repo'
    details = {"mode": "nl2repo"}
    
    metadata = make_metadata(
        llm_config,
        dataset_description,
        'CodeActAgent',
        args.max_iterations,
        "",
        args.eval_output_dir,
        details=details
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(
        nl2repo_tests, 
        output_file,
        args.eval_n_limit,
    )
    
    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=12 * 60 * 60,  # 12 hours per instance
        max_retries=3,
    )