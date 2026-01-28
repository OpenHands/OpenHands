import copy
import docker
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    get_default_sandbox_config_for_eval,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.core.config import (
    AppConfig,
    LLMConfig,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger

NL2REPO_IMAGE_REGISTRY = os.environ.get(
    'NL2REPO_IMAGE_REGISTRY',
    'ghcr.io/multimodal-art-projection/nl2repobench'
)

@dataclass
class TestResult:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total: int = 0
    success_rate: float = 0.0
    test_output: str = ""
    command_results: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.command_results is None:
            self.command_results = []


def get_instance_test_image(instance_id: str) -> str:
    project_name = instance_id.lower()
    return f'{NL2REPO_IMAGE_REGISTRY}/{project_name}:1.0'


def remove_package_files(workspace_path: str) -> None:
    package_files = [
        "setup.py",
        "pyproject.toml",
        "setup.cfg",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "tox.ini",
        "pytest.ini",
        "poetry.lock",
        "Pipfile",
        "Pipfile.lock",
        "environment.yml",
        "conda-env.yaml",
        "manifest.in",
        "MANIFEST.in"
    ]
    for root, _, files in os.walk(workspace_path):
        for file in files:
            if file in package_files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logger.info(f'Removed package file: {file_path}')
                except Exception as e:
                    logger.warning(f'Failed to remove {file_path}: {str(e)}')


def remove_test_files(workspace_path: str, test_files: List[str]) -> None:
    for test_file in test_files:
        target_path = os.path.join(workspace_path, test_file)
        try:
            if os.path.exists(target_path):
                if os.path.isdir(target_path):
                    shutil.rmtree(target_path)
                    logger.info(f'Removed test directory: {target_path}')
                else:
                    os.remove(target_path)
                    logger.info(f'Removed test file: {target_path}')
        except Exception as e:
            logger.error(f'Failed to remove {target_path}: {str(e)}')


def analyze_pytest_results(output: str, total_test_cases: int = 0) -> Dict[str, Any]:
    results = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'total': total_test_cases,
        'success_rate': 0.0
    }

    for line in output.split('\n'):
        # Match "X passed"
        passed_match = re.search(r'(\d+) passed', line)
        if passed_match:
            results['passed'] = int(passed_match.group(1))
        
        # Match "X failed"
        failed_match = re.search(r'(\d+) failed', line)
        if failed_match:
            results['failed'] = int(failed_match.group(1))

        # Match "X error"
        error_match = re.search(r'(\d+) error', line)
        if error_match:
            results['errors'] = int(error_match.group(1))
    
    results['success_rate'] = min(results['passed'] / results['total'], 1.0)
    
    return results


def get_config(metadata: EvalMetadata, instance: pd.Series) -> AppConfig:
    base_container_image = get_instance_test_image(instance['instance_id'])
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image    
    config = AppConfig(
        run_as_openhands=False,
        runtime="docker",
        sandbox=sandbox_config,
        workspace_base=None,
        workspace_mount_path=None,
    )
    return config


def run_command_in_docker(
    container,
    command: str,
    timeout: int = 1800,
) -> tuple[str, int]:
    result_holder = {'output': '', 'exit_code': -1, 'error': None}
    
    def execute():
        try:
            result = container.exec_run(
                cmd=f'bash -c "{command}"',
                stdout=True,
                stderr=True,
            )
            result_holder['output'] = result.output.decode('utf-8', errors='replace')
            result_holder['exit_code'] = result.exit_code
        except Exception as e:
            result_holder['error'] = str(e)
            logger.error(f'Error running command in docker: {e}')
    
    thread = threading.Thread(target=execute)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        logger.warning(f'Command timed out after {timeout} seconds: {command}')
        return f'Command timed out after {timeout} seconds', 124
    
    if result_holder['error']:
        return result_holder['error'], -1
    
    return result_holder['output'], result_holder['exit_code']


def run_tests_in_container(
    container,
    test_commands: List[str],
    test_case_count: int = 0,
    output_log_file: str | None = None,
) -> TestResult:
    result = TestResult(total=test_case_count)
    all_output = []
    command_results = []
    
    for i, command in enumerate(test_commands):
        logger.info(f'Executing command {i + 1}/{len(test_commands)}: {command}')
        
        output, exit_code = run_command_in_docker(container, command, timeout=1800)
        
        logger.info(f'Command exit code: {exit_code}')
        logger.info(f'Command output:\n{output}')
        all_output.append(output)
        
        if output_log_file:
            log_dir_path = os.path.dirname(output_log_file)
            if log_dir_path:
                os.makedirs(log_dir_path, exist_ok=True)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            with open(output_log_file, 'a', encoding='utf-8') as f:
                f.write(f'\n{"=" * 80}\n')
                f.write(f'[{timestamp}] Command {i + 1}/{len(test_commands)}: {command}\n')
                f.write(f'Exit Code: {exit_code}\n')
                f.write(f'{"=" * 80}\n')
                f.write(output)
                if not output.endswith('\n'):
                    f.write('\n')
        
        command_results.append({
            'command': command,
            'exit_code': exit_code,
            'output': output,
            'timestamp': datetime.now().isoformat(),
        })
    
    result.test_output = '\n'.join(all_output)
    result.command_results = command_results
    
    pytest_results = analyze_pytest_results(result.test_output, test_case_count)
    result.passed = pytest_results['passed']
    result.failed = pytest_results['failed']
    result.errors = pytest_results['errors']
    result.total = pytest_results['total']
    result.success_rate = pytest_results['success_rate']
    
    return result


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    log_dir: str | None = None,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    if reset_logger:
        assert log_dir is not None, "Can't reset logger without a provided log directory."
        os.makedirs(log_dir, exist_ok=True)
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    config = get_config(metadata, instance)
    instance_id = instance.instance_id
    
    if 'test_result' not in instance.keys():
        instance['test_result'] = {}
    
    instance['test_result']['eval_report'] = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'total': 0,
        'success_rate': 0.0,
        'error_eval': False,
    }

    workspace_dir = instance.get('workspace_dir')
    test_commands = instance.get('test_commands', [])
    test_case_count = instance.get('test_case_count', 0)
    test_files_to_remove = instance.get('test_files', [])

    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = 1
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance_id}, '
            f'setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )
    
    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count
    metadata.details['remote_runtime_resource_factor'] = config.sandbox.remote_runtime_resource_factor

    try:
        docker_client = docker.from_env(timeout=3600)
        base_container_image = get_instance_test_image(instance_id)
        
        logger.info(f'[{instance_id}] Starting docker container with image: {base_container_image}')
        container = docker_client.containers.run(
            base_container_image,
            command='tail -f /dev/null',
            detach=True,
            remove=False,
        )
        container_id = container.id[:12]
        logger.info(f'[{instance_id}] Container {container_id} started successfully')
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_workspace = os.path.join(temp_dir, 'workspace')
                shutil.copytree(workspace_dir, temp_workspace)
                remove_package_files(temp_workspace)
                remove_test_files(temp_workspace, test_files_to_remove)
                
                output, exit_code = run_command_in_docker(
                    container, 'mkdir -p /tmp/generated_code'
                )
                if exit_code != 0:
                    logger.error(f'[{instance_id}] Failed to create /tmp/generated_code: {output}')
                    eval_report = instance['test_result']['eval_report']
                    eval_report['error_eval'] = True
                    return EvalOutput(
                        instance_id=instance_id,
                        test_result={
                            'eval_report': eval_report,
                            'test_output': output,
                        },
                        metadata=metadata,
                    )
                
                logger.info(f'[{instance_id}] Copying workspace to container...')
                try:
                    subprocess.run(
                        f'cd {temp_dir} && tar -czf workspace.tar workspace && '
                        f'docker cp workspace.tar {container_id}:/tmp/generated_code/ && '
                        f'docker exec {container_id} sh -c "cd /tmp/generated_code && tar -xzf workspace.tar"',
                        shell=True,
                        check=True,
                    )
                except Exception as e:
                    logger.error(f'[{instance_id}] Failed to copy workspace to container: {e}')
                    eval_report = instance['test_result']['eval_report']
                    eval_report['error_eval'] = True
                    return EvalOutput(
                        instance_id=instance_id,
                        test_result={
                            'eval_report': eval_report,
                            'test_output': str(e),
                        },
                        metadata=metadata,
                    )
                
                output, exit_code = run_command_in_docker(
                    container,
                    'cp -r /tmp/generated_code/workspace/* /workspace/ && rm -rf /tmp/generated_code'
                )
                if exit_code != 0:
                    logger.error(f'[{instance_id}] Failed to merge generated code into /workspace: {output}')
                    eval_report = instance['test_result']['eval_report']
                    eval_report['error_eval'] = True
                    return EvalOutput(
                        instance_id=instance_id,
                        test_result={
                            'eval_report': eval_report,
                            'test_output': output,
                        },
                        metadata=metadata,
                    )
            
            logger.info(f'[{instance_id}] Running {len(test_commands)} test commands...')
            output_log_file = os.path.join(log_dir, f'{instance_id}_output.log') if log_dir else None
            test_result = run_tests_in_container(
                container, 
                test_commands, 
                test_case_count,
                output_log_file=output_log_file,
            )
        finally:
            container.remove(force=True)
            logger.info(f'[{instance_id}] Container {container_id} removed')
        
        eval_report = {
            'passed': test_result.passed,
            'failed': test_result.failed,
            'errors': test_result.errors,
            'total': test_result.total,
            'success_rate': test_result.success_rate,
            'command_results': test_result.command_results,
            'timestamp': datetime.now().isoformat(),
        }
        
        test_output = test_result.test_output
        
        logger.info(
            f'[{instance_id}] Evaluation complete: '
            f'{test_result.passed}/{test_result.total} tests passed '
            f'({test_result.success_rate:.2%})'
        )
        
        return EvalOutput(
            instance_id=instance_id,
            test_result={
                'eval_report': eval_report,
                'test_output': test_output,
            },
            metadata=metadata,
        )
        
    except Exception as e:
        logger.error(f'[{instance_id}] Evaluation error: {e}')
        eval_report = instance['test_result']['eval_report']
        eval_report['error_eval'] = True
        return EvalOutput(
            instance_id=instance_id,
            test_result={
                'eval_report': eval_report,
                'test_output': str(e),
            },
            metadata=metadata,
        )


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='/workspace/swe-data/dataset/nl2repo/nl2repo.jsonl',
        help='Path to NL2Repo dataset file',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    args, _ = parser.parse_known_args()

    assert os.path.isdir(args.eval_output_dir), f'eval_output_dir must be a directory: {args.eval_output_dir}'
    assert os.path.isfile(args.dataset), f'dataset file not found: {args.dataset}'
    
    with open(args.dataset) as f:
        dataset = pd.DataFrame.from_records(
            [json.loads(line) for line in tqdm(f, desc='Loading dataset')]
        )
    
    assert 'instance_id' in dataset.columns, 'Dataset file must contain instance_id column.'
    logger.info(f'Loaded {len(dataset)} instances from {args.dataset}')

    eval_output_dir = args.eval_output_dir
    instances_with_results = []
    
    instances_dir = os.path.join(eval_output_dir, 'instances')
    if os.path.exists(instances_dir):
        for instance_id in os.listdir(instances_dir):
            workspace_dir = os.path.join(instances_dir, instance_id)
            if os.path.isdir(workspace_dir):
                instances_with_results.append(instance_id)
    
    logger.info(f'Found {len(instances_with_results)} instances with generated results')
    
    dataset = dataset[dataset['instance_id'].isin(instances_with_results)].reset_index(drop=True)
    logger.info(f'Filtering dataset: {len(dataset)} instances to evaluate')
    
    if len(dataset) == 0:
        logger.error(f'No instances with results found in {eval_output_dir}')
        logger.error(f'Expected structure: {eval_output_dir}/instances/{{instance_id}}/')
        exit(1)
    
    for idx, row in dataset.iterrows():
        instance_id = row['instance_id']
        workspace_dir = os.path.join(eval_output_dir, 'instances', instance_id)
        dataset.at[idx, 'workspace_dir'] = workspace_dir

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(eval_output_dir, f'output.eval.{timestamp}.jsonl')
    log_dir_path = output_file.replace('.jsonl', '.logs')
    instances = prepare_dataset(dataset, output_file, args.eval_n_limit)

    metadata = EvalMetadata(
        agent_class='dummy_agent',
        llm_config=LLMConfig(model='dummy_model'),
        max_iterations=1,
        eval_output_dir=eval_output_dir,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        .decode('utf-8').strip(),
        dataset='nl2repo',
        details={},
    )

    process_instance_func = partial(
        process_instance,
        log_dir=log_dir_path,
    )

    run_evaluation(
        instances,
        metadata=metadata,
        output_file=output_file,
        num_workers=args.eval_num_workers,
        process_instance_func=process_instance_func,
    )

    evaluated_predictions = pd.read_json(output_file, lines=True)
    
    def get_passed(row):
        return row.get('test_result', {}).get('eval_report', {}).get('passed', 0)
    
    def get_total(row):
        return row.get('test_result', {}).get('eval_report', {}).get('total', 0)
    
    def get_success_rate(row):
        return row.get('test_result', {}).get('eval_report', {}).get('success_rate', 0.0)
    
    total_passed = evaluated_predictions.apply(get_passed, axis=1).sum()
    total_tests = evaluated_predictions.apply(get_total, axis=1).sum()
    avg_success_rate = evaluated_predictions.apply(get_success_rate, axis=1).mean()
    instance_success_rates = evaluated_predictions.apply(get_success_rate, axis=1)

    instance_metrics = []
    for idx, row in evaluated_predictions.iterrows():
        instance_id = row.get('instance_id', f'instance_{idx}')
        success_rate = get_success_rate(row)
        passed = get_passed(row)
        total = get_total(row)
        instance_metrics.append({
            'instance_id': instance_id,
            'success_rate': float(success_rate),
            'tests_passed': int(passed),
            'total_tests': int(total),
        })
    
    eval_summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'nl2repo',
        'total_instances_evaluated': len(evaluated_predictions),
        'total_tests_passed': int(total_passed),
        'total_tests': int(total_tests),
        'average_success_rate': float(avg_success_rate),
        'instances': instance_metrics,
    }
    
    summary_file = os.path.join(eval_output_dir, f'eval_summary.{timestamp}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    logger.info(f'Evaluation summary saved to: {summary_file}')
    
    logger.info('=' * 60)
    logger.info('NL2Repo Evaluation Summary')
    logger.info('=' * 60)
    logger.info(f'Total instances evaluated: {len(evaluated_predictions)}')
    logger.info(f'Total tests passed: {total_passed}/{total_tests}')
    logger.info(f'Average success rate: {avg_success_rate:.2%}')
    logger.info('=' * 60)