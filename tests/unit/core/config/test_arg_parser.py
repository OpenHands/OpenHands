import pytest

from openhands.core.config import (
    get_evaluation_parser,
    get_headless_parser,
)


def test_headless_parser_default_values():
    parser = get_headless_parser()
    args = parser.parse_args([])

    assert args.directory is None
    assert args.task == ''
    assert args.file is None
    assert args.agent_cls is None
    assert args.max_iterations is None
    assert args.max_budget_per_task is None
    assert args.name == ''
    assert not args.no_auto_continue
    assert args.selected_repo is None


def test_evaluation_parser_default_values():
    parser = get_evaluation_parser()
    args = parser.parse_args([])

    assert args.directory is None
    assert args.task == ''
    assert args.file is None
    assert args.agent_cls is None
    assert args.max_iterations is None
    assert args.max_budget_per_task is None
    assert args.eval_output_dir == 'evaluation/evaluation_outputs/outputs'
    assert args.eval_n_limit is None
    assert args.eval_num_workers == 4
    assert args.eval_note is None
    assert args.name == ''
    assert not args.no_auto_continue
    assert args.selected_repo is None


def test_evaluation_parser_custom_values():
    parser = get_evaluation_parser()
    args = parser.parse_args(
        [
            '-v',
            '-d',
            '/path/to/dir',
            '-t',
            'custom task',
            '-f',
            'task.txt',
            '-c',
            'CustomAgent',
            '-i',
            '50',
            '-b',
            '100.5',
            '--eval-output-dir',
            'custom/output',
            '--eval-n-limit',
            '10',
            '--eval-num-workers',
            '8',
            '--eval-note',
            'Test run',
            '-n',
            'test_session',
            '--no-auto-continue',
            '--selected-repo',
            'owner/repo',
        ]
    )

    assert args.directory == '/path/to/dir'
    assert args.task == 'custom task'
    assert args.file == 'task.txt'
    assert args.agent_cls == 'CustomAgent'
    assert args.max_iterations == 50
    assert args.max_budget_per_task == pytest.approx(100.5)
    assert args.eval_output_dir == 'custom/output'
    assert args.eval_n_limit == 10
    assert args.eval_num_workers == 8
    assert args.eval_note == 'Test run'
    assert args.name == 'test_session'
    assert args.no_auto_continue
    assert args.version
    assert args.selected_repo == 'owner/repo'


def test_parser_file_overrides_task():
    parser = get_headless_parser()
    args = parser.parse_args(['-t', 'task from command', '-f', 'task_file.txt'])

    assert args.task == 'task from command'
    assert args.file == 'task_file.txt'


def test_parser_invalid_max_iterations():
    parser = get_headless_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['-i', 'not_a_number'])


def test_parser_invalid_max_budget():
    parser = get_headless_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['-b', 'not_a_number'])


def test_evaluation_parser_invalid_eval_n_limit():
    parser = get_evaluation_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--eval-n-limit', 'not_a_number'])


def test_evaluation_parser_invalid_eval_num_workers():
    parser = get_evaluation_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--eval-num-workers', 'not_a_number'])


def test_headless_parser_help_message(capsys):
    parser = get_headless_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--help'])
    captured = capsys.readouterr()
    help_output = captured.out
    print(help_output)
    expected_elements = [
        'usage:',
        'Run the agent via CLI',
        'options:',
        '-v, --version',
        '-h, --help',
        '-d DIRECTORY, --directory DIRECTORY',
        '-t TASK, --task TASK',
        '-f FILE, --file FILE',
        '-c AGENT_CLS, --agent-cls AGENT_CLS',
        '-i MAX_ITERATIONS, --max-iterations MAX_ITERATIONS',
        '-b MAX_BUDGET_PER_TASK, --max-budget-per-task MAX_BUDGET_PER_TASK',
        '--agent-config AGENT_CONFIG',
        '-n NAME, --name NAME',
        '--config-file CONFIG_FILE',
        '--no-auto-continue',
        '--selected-repo SELECTED_REPO',
        '--log-level LOG_LEVEL',
    ]

    for element in expected_elements:
        assert element in help_output, f"Expected '{element}' to be in the help message"

    option_count = help_output.count('  -')
    assert option_count == 14, f'Expected 14 options, found {option_count}'


def test_evaluation_parser_help_message(capsys):
    parser = get_evaluation_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--help'])
    captured = capsys.readouterr()
    help_output = captured.out
    print(help_output)
    expected_elements = [
        'usage:',
        'Run OpenHands in evaluation mode',
        'options:',
        '-v, --version',
        '-h, --help',
        '-d DIRECTORY, --directory DIRECTORY',
        '-t TASK, --task TASK',
        '-f FILE, --file FILE',
        '-c AGENT_CLS, --agent-cls AGENT_CLS',
        '-i MAX_ITERATIONS, --max-iterations MAX_ITERATIONS',
        '-b MAX_BUDGET_PER_TASK, --max-budget-per-task MAX_BUDGET_PER_TASK',
        '--eval-output-dir EVAL_OUTPUT_DIR',
        '--eval-n-limit EVAL_N_LIMIT',
        '--eval-num-workers EVAL_NUM_WORKERS',
        '--eval-note EVAL_NOTE',
        '--eval-ids EVAL_IDS',
        '--agent-config AGENT_CONFIG',
        '-n NAME, --name NAME',
        '--config-file CONFIG_FILE',
        '--no-auto-continue',
        '--selected-repo SELECTED_REPO',
        '--log-level LOG_LEVEL',
    ]

    for element in expected_elements:
        assert element in help_output, f"Expected '{element}' to be in the help message"

    option_count = help_output.count('  -')
    assert option_count == 19, f'Expected 19 options, found {option_count}'


def test_selected_repo_format():
    """Test that the selected-repo argument accepts owner/repo format."""
    parser = get_headless_parser()
    args = parser.parse_args(['--selected-repo', 'owner/repo'])
    assert args.selected_repo == 'owner/repo'
