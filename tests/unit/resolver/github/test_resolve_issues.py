import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.core.config import LLMConfig
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import (
    CmdOutputMetadata,
    CmdOutputObservation,
    NullObservation,
)
from openhands.integrations.service_types import ProviderType
from openhands.llm.llm import LLM
from openhands.resolver.interfaces.github import GithubIssueHandler, GithubPRHandler
from openhands.resolver.interfaces.issue import Issue, ReviewThread
from openhands.resolver.interfaces.issue_definitions import (
    ServiceContextIssue,
    ServiceContextPR,
)
from openhands.resolver.issue_resolver import IssueResolver
from openhands.resolver.resolver_output import ResolverOutput


@pytest.fixture
def default_mock_args():
    """Fixture that provides a default mock args object with common values.

    Tests can override specific attributes as needed.
    """
    mock_args = MagicMock()
    mock_args.selected_repo = 'test-owner/test-repo'
    mock_args.token = 'test-token'
    mock_args.username = 'test-user'
    mock_args.max_iterations = 5
    mock_args.output_dir = '/tmp'
    mock_args.llm_model = 'test'
    mock_args.llm_api_key = 'test'
    mock_args.llm_base_url = None
    mock_args.base_domain = None
    mock_args.runtime_container_image = None
    mock_args.base_container_image = None
    mock_args.is_experimental = False
    mock_args.issue_number = None
    mock_args.comment_id = None
    mock_args.repo_instruction_file = None
    mock_args.issue_type = 'issue'
    mock_args.prompt_file = None
    return mock_args


@pytest.fixture
def mock_github_token():
    """Fixture that patches the identify_token function to return GitHub provider type.

    This eliminates the need for repeated patching in each test function.
    """
    with patch(
        'openhands.resolver.issue_resolver.identify_token',
        return_value=ProviderType.GITHUB,
    ) as patched:
        yield patched


@pytest.fixture
def mock_output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = os.path.join(temp_dir, 'repo')
        # Initialize a GitHub repo in "repo" and add a commit with "README.md"
        os.makedirs(repo_path)
        os.system(f'git init {repo_path}')
        readme_path = os.path.join(repo_path, 'README.md')
        with open(readme_path, 'w') as f:
            f.write('hello world')
        os.system(f'git -C {repo_path} add README.md')
        os.system(f"git -C {repo_path} commit -m 'Initial commit'")
        yield temp_dir


@pytest.fixture
def mock_subprocess():
    with patch('subprocess.check_output') as mock_check_output:
        yield mock_check_output


@pytest.fixture
def mock_os():
    with patch('os.system') as mock_system, patch('os.path.join') as mock_join:
        yield mock_system, mock_join


@pytest.fixture
def mock_user_instructions_template():
    return 'Issue: {{ body }}\n\nPlease fix this issue.'


@pytest.fixture
def mock_conversation_instructions_template():
    return 'Instructions: {{ repo_instruction }}'


@pytest.fixture
def mock_followup_prompt_template():
    return 'Issue context: {{ issues }}\n\nReview comments: {{ review_comments }}\n\nReview threads: {{ review_threads }}\n\nFiles: {{ files }}\n\nThread comments: {{ thread_context }}\n\nPlease fix this issue.'


def create_cmd_output(exit_code: int, content: str, command: str):
    return CmdOutputObservation(
        content=content,
        command=command,
        metadata=CmdOutputMetadata(exit_code=exit_code),
    )


def test_initialize_runtime(default_mock_args, mock_github_token):
    mock_runtime = MagicMock()
    mock_runtime.run_action.side_effect = [
        create_cmd_output(exit_code=0, content='', command='cd /workspace'),
        create_cmd_output(
            exit_code=0, content='', command='git config --global core.pager ""'
        ),
    ]

    # Create resolver with mocked token identification
    resolver = IssueResolver(default_mock_args)

    resolver.initialize_runtime(mock_runtime)

    assert mock_runtime.run_action.call_count == 2
    mock_runtime.run_action.assert_any_call(CmdRunAction(command='cd /workspace'))
    mock_runtime.run_action.assert_any_call(
        CmdRunAction(command='git config --global core.pager ""')
    )


@pytest.mark.asyncio
async def test_resolve_issue_no_issues_found(default_mock_args, mock_github_token):
    """Test the resolve_issue method when no issues are found."""
    # Mock dependencies
    mock_handler = MagicMock()
    mock_handler.get_converted_issues.return_value = []  # Return empty list

    # Customize the mock args for this test
    default_mock_args.issue_number = 5432

    # Create a resolver instance with mocked token identification
    resolver = IssueResolver(default_mock_args)

    # Mock the issue handler
    resolver.issue_handler = mock_handler

    # Test that the correct exception is raised
    with pytest.raises(ValueError) as exc_info:
        await resolver.resolve_issue()

    # Verify the error message
    assert 'No issues found for issue number 5432' in str(exc_info.value)
    assert 'test-owner/test-repo' in str(exc_info.value)

    mock_handler.get_converted_issues.assert_called_once_with(
        issue_numbers=[5432], comment_id=None
    )


def test_download_issues_from_github():
    llm_config = LLMConfig(model='test', api_key='test')
    handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), llm_config
    )

    mock_issues_response = MagicMock()
    mock_issues_response.json.side_effect = [
        [
            {'number': 1, 'title': 'Issue 1', 'body': 'This is an issue'},
            {
                'number': 2,
                'title': 'PR 1',
                'body': 'This is a pull request',
                'pull_request': {},
            },
            {'number': 3, 'title': 'Issue 2', 'body': 'This is another issue'},
        ],
        None,
    ]
    mock_issues_response.raise_for_status = MagicMock()

    mock_comments_response = MagicMock()
    mock_comments_response.json.return_value = []
    mock_comments_response.raise_for_status = MagicMock()

    def get_mock_response(url, *args, **kwargs):
        if '/comments' in url:
            return mock_comments_response
        return mock_issues_response

    with patch('httpx.get', side_effect=get_mock_response):
        issues = handler.get_converted_issues(issue_numbers=[1, 3])

    assert len(issues) == 2
    assert handler.issue_type == 'issue'
    assert all(isinstance(issue, Issue) for issue in issues)
    assert [issue.number for issue in issues] == [1, 3]
    assert [issue.title for issue in issues] == ['Issue 1', 'Issue 2']
    assert [issue.review_comments for issue in issues] == [None, None]
    assert [issue.closing_issues for issue in issues] == [None, None]
    assert [issue.thread_ids for issue in issues] == [None, None]


def test_download_pr_from_github():
    llm_config = LLMConfig(model='test', api_key='test')
    handler = ServiceContextPR(GithubPRHandler('owner', 'repo', 'token'), llm_config)
    mock_pr_response = MagicMock()
    mock_pr_response.json.side_effect = [
        [
            {
                'number': 1,
                'title': 'PR 1',
                'body': 'This is a pull request',
                'head': {'ref': 'b1'},
            },
            {
                'number': 2,
                'title': 'My PR',
                'body': 'This is another pull request',
                'head': {'ref': 'b2'},
            },
            {'number': 3, 'title': 'PR 3', 'body': 'Final PR', 'head': {'ref': 'b3'}},
        ],
        None,
    ]
    mock_pr_response.raise_for_status = MagicMock()

    # Mock for PR comments response
    mock_comments_response = MagicMock()
    mock_comments_response.json.return_value = []  # No PR comments
    mock_comments_response.raise_for_status = MagicMock()

    # Mock for GraphQL request (for download_pr_metadata)
    mock_graphql_response = MagicMock()
    mock_graphql_response.json.side_effect = lambda: {
        'data': {
            'repository': {
                'pullRequest': {
                    'closingIssuesReferences': {
                        'edges': [
                            {'node': {'body': 'Issue 1 body', 'number': 1}},
                            {'node': {'body': 'Issue 2 body', 'number': 2}},
                        ]
                    },
                    'reviewThreads': {
                        'edges': [
                            {
                                'node': {
                                    'isResolved': False,
                                    'id': '1',
                                    'comments': {
                                        'nodes': [
                                            {
                                                'body': 'Unresolved comment 1',
                                                'path': '/frontend/header.tsx',
                                            },
                                            {'body': 'Follow up thread'},
                                        ]
                                    },
                                }
                            },
                            {
                                'node': {
                                    'isResolved': True,
                                    'id': '2',
                                    'comments': {
                                        'nodes': [
                                            {
                                                'body': 'Resolved comment 1',
                                                'path': '/some/file.py',
                                            }
                                        ]
                                    },
                                }
                            },
                            {
                                'node': {
                                    'isResolved': False,
                                    'id': '3',
                                    'comments': {
                                        'nodes': [
                                            {
                                                'body': 'Unresolved comment 3',
                                                'path': '/another/file.py',
                                            }
                                        ]
                                    },
                                }
                            },
                        ]
                    },
                }
            }
        }
    }

    mock_graphql_response.raise_for_status = MagicMock()

    def get_mock_response(url, *args, **kwargs):
        if '/comments' in url:
            return mock_comments_response
        return mock_pr_response

    with patch('httpx.get', side_effect=get_mock_response):
        with patch('httpx.post', return_value=mock_graphql_response):
            issues = handler.get_converted_issues(issue_numbers=[1, 2, 3])

    assert len(issues) == 3
    assert handler.issue_type == 'pr'
    assert all(isinstance(issue, Issue) for issue in issues)
    assert [issue.number for issue in issues] == [1, 2, 3]
    assert [issue.title for issue in issues] == ['PR 1', 'My PR', 'PR 3']
    assert [issue.head_branch for issue in issues] == ['b1', 'b2', 'b3']

    assert len(issues[0].review_threads) == 2  # Only unresolved threads
    assert (
        issues[0].review_threads[0].comment
        == 'Unresolved comment 1\n---\nlatest feedback:\nFollow up thread\n'
    )
    assert issues[0].review_threads[0].files == ['/frontend/header.tsx']
    assert (
        issues[0].review_threads[1].comment
        == 'latest feedback:\nUnresolved comment 3\n'
    )
    assert issues[0].review_threads[1].files == ['/another/file.py']
    assert issues[0].closing_issues == ['Issue 1 body', 'Issue 2 body']
    assert issues[0].thread_ids == ['1', '3']


@pytest.mark.asyncio
async def test_complete_runtime(default_mock_args, mock_github_token):
    """Test the complete_runtime method."""
    mock_runtime = MagicMock()
    mock_runtime.run_action.side_effect = [
        create_cmd_output(exit_code=0, content='', command='cd /workspace'),
        create_cmd_output(
            exit_code=0, content='', command='git config --global core.pager ""'
        ),
        create_cmd_output(
            exit_code=0,
            content='',
            command='git config --global --add safe.directory /workspace',
        ),
        create_cmd_output(
            exit_code=0, content='', command='git diff base_commit_hash fix'
        ),
        create_cmd_output(exit_code=0, content='git diff content', command='git apply'),
    ]

    # Create resolver with mocked token identification
    resolver = IssueResolver(default_mock_args)

    result = await resolver.complete_runtime(mock_runtime, 'base_commit_hash')

    assert result == {'git_patch': 'git diff content'}
    assert mock_runtime.run_action.call_count == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'test_case',
    [
        {
            'name': 'successful_run',
            'run_controller_return': MagicMock(
                history=[NullObservation(content='')],
                metrics=MagicMock(
                    get=MagicMock(return_value={'test_result': 'passed'})
                ),
                last_error=None,
            ),
            'run_controller_raises': None,
            'expected_success': True,
            'expected_error': None,
            'expected_explanation': 'Issue resolved successfully',
            'is_pr': False,
            'comment_success': None,
        },
        {
            'name': 'value_error',
            'run_controller_return': None,
            'run_controller_raises': ValueError('Test value error'),
            'expected_success': False,
            'expected_error': 'Agent failed to run or crashed',
            'expected_explanation': 'Agent failed to run',
            'is_pr': False,
            'comment_success': None,
        },
        {
            'name': 'runtime_error',
            'run_controller_return': None,
            'run_controller_raises': RuntimeError('Test runtime error'),
            'expected_success': False,
            'expected_error': 'Agent failed to run or crashed',
            'expected_explanation': 'Agent failed to run',
            'is_pr': False,
            'comment_success': None,
        },
        {
            'name': 'json_decode_error',
            'run_controller_return': MagicMock(
                history=[NullObservation(content='')],
                metrics=MagicMock(
                    get=MagicMock(return_value={'test_result': 'passed'})
                ),
                last_error=None,
            ),
            'run_controller_raises': None,
            'expected_success': True,
            'expected_error': None,
            'expected_explanation': 'Non-JSON explanation',
            'is_pr': True,
            'comment_success': [
                True,
                False,
            ],  # To trigger the PR success logging code path
        },
    ],
)
async def test_process_issue(
    default_mock_args,
    mock_github_token,
    mock_output_dir,
    mock_user_instructions_template,
    test_case,
):
    """Test the process_issue method with different scenarios."""
    # Set up test data
    issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=1,
        title='Test Issue',
        body='This is a test issue',
    )
    base_commit = 'abcdef1234567890'

    # Customize the mock args for this test
    default_mock_args.output_dir = mock_output_dir
    default_mock_args.issue_type = 'pr' if test_case.get('is_pr', False) else 'issue'

    # Create a resolver instance with mocked token identification
    resolver = IssueResolver(default_mock_args)
    resolver.user_instructions_prompt_template = mock_user_instructions_template

    # Mock the handler with LLM config
    llm_config = LLMConfig(model='test', api_key='test')
    handler_instance = MagicMock()
    handler_instance.guess_success.return_value = (
        test_case['expected_success'],
        test_case.get('comment_success', None),
        test_case['expected_explanation'],
    )
    handler_instance.get_instruction.return_value = (
        'Test instruction',
        'Test conversation instructions',
        [],
    )
    handler_instance.issue_type = 'pr' if test_case.get('is_pr', False) else 'issue'
    handler_instance.llm = LLM(llm_config, service_id='test-service')

    # Mock the runtime and its methods
    mock_runtime = MagicMock()
    mock_runtime.connect = AsyncMock()
    mock_runtime.run_action.return_value = CmdOutputObservation(
        content='test patch',
        command='git diff',
        metadata=CmdOutputMetadata(exit_code=0),
    )
    mock_runtime.event_stream.subscribe = MagicMock()

    # Mock the create_runtime function
    mock_create_runtime = MagicMock(return_value=mock_runtime)

    # Mock the run_controller function
    mock_run_controller = AsyncMock()
    if test_case['run_controller_raises']:
        mock_run_controller.side_effect = test_case['run_controller_raises']
    else:
        mock_run_controller.return_value = test_case['run_controller_return']

    # Patch the necessary functions and methods
    with (
        patch('openhands.resolver.issue_resolver.create_runtime', mock_create_runtime),
        patch('openhands.resolver.issue_resolver.run_controller', mock_run_controller),
        patch.object(
            resolver, 'complete_runtime', return_value={'git_patch': 'test patch'}
        ),
        patch.object(resolver, 'initialize_runtime') as mock_initialize_runtime,
    ):
        # Call the process_issue method
        result = await resolver.process_issue(issue, base_commit, handler_instance)

        # Assert the result matches our expectations
        assert isinstance(result, ResolverOutput)
        assert result.issue == issue
        assert result.base_commit == base_commit
        assert result.git_patch == 'test patch'
        assert result.success == test_case['expected_success']
        assert result.result_explanation == test_case['expected_explanation']
        assert result.error == test_case['expected_error']

        # Assert that the mocked functions were called
        mock_create_runtime.assert_called_once()
        mock_runtime.connect.assert_called_once()
        mock_initialize_runtime.assert_called_once()
        mock_run_controller.assert_called_once()
        resolver.complete_runtime.assert_awaited_once_with(mock_runtime, base_commit)

        # Assert run_controller was called with the right parameters
        if not test_case['run_controller_raises']:
            # Check that the first positional argument is a config
            assert 'config' in mock_run_controller.call_args[1]
            # Check that initial_user_action is a MessageAction with the right content
            assert isinstance(
                mock_run_controller.call_args[1]['initial_user_action'], MessageAction
            )
            assert mock_run_controller.call_args[1]['runtime'] == mock_runtime

        # Assert that guess_success was called only for successful runs
        if test_case['expected_success']:
            handler_instance.guess_success.assert_called_once()
        else:
            handler_instance.guess_success.assert_not_called()


def test_get_instruction(
    mock_user_instructions_template,
    mock_conversation_instructions_template,
    mock_followup_prompt_template,
):
    issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=123,
        title='Test Issue',
        body='This is a test issue refer to image ![First Image](https://sampleimage.com/image1.png)',
    )
    mock_llm_config = LLMConfig(model='test_model', api_key='test_api_key')
    issue_handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), mock_llm_config
    )
    instruction, conversation_instructions, images_urls = issue_handler.get_instruction(
        issue,
        mock_user_instructions_template,
        mock_conversation_instructions_template,
        None,
    )
    expected_instruction = 'Issue: Test Issue\n\nThis is a test issue refer to image ![First Image](https://sampleimage.com/image1.png)\n\nPlease fix this issue.'

    assert images_urls == ['https://sampleimage.com/image1.png']
    assert issue_handler.issue_type == 'issue'
    assert instruction == expected_instruction
    assert conversation_instructions is not None

    issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=123,
        title='Test Issue',
        body='This is a test issue',
        closing_issues=['Issue 1 fix the type'],
        review_threads=[
            ReviewThread(
                comment="There is still a typo 'pthon' instead of 'python'", files=[]
            )
        ],
        thread_comments=[
            "I've left review comments, please address them",
            'This is a valid concern.',
        ],
    )

    pr_handler = ServiceContextPR(
        GithubPRHandler('owner', 'repo', 'token'), mock_llm_config
    )
    instruction, conversation_instructions, images_urls = pr_handler.get_instruction(
        issue,
        mock_followup_prompt_template,
        mock_conversation_instructions_template,
        None,
    )
    expected_instruction = "Issue context: [\n    \"Issue 1 fix the type\"\n]\n\nReview comments: None\n\nReview threads: [\n    \"There is still a typo 'pthon' instead of 'python'\"\n]\n\nFiles: []\n\nThread comments: I've left review comments, please address them\n---\nThis is a valid concern.\n\nPlease fix this issue."

    assert images_urls == []
    assert pr_handler.issue_type == 'pr'
    # Compare content ignoring exact formatting
    assert "There is still a typo 'pthon' instead of 'python'" in instruction
    assert "I've left review comments, please address them" in instruction
    assert 'This is a valid concern' in instruction
    assert conversation_instructions is not None


def test_file_instruction():
    issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=123,
        title='Test Issue',
        body='This is a test issue ![image](https://sampleimage.com/sample.png)',
    )
    # load prompt from openhands/resolver/prompts/resolve/basic.jinja
    with open('openhands/resolver/prompts/resolve/basic.jinja', 'r') as f:
        prompt = f.read()

    with open(
        'openhands/resolver/prompts/resolve/basic-conversation-instructions.jinja', 'r'
    ) as f:
        conversation_instructions_template = f.read()

    # Test without thread comments
    mock_llm_config = LLMConfig(model='test_model', api_key='test_api_key')
    issue_handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), mock_llm_config
    )
    instruction, conversation_instructions, images_urls = issue_handler.get_instruction(
        issue, prompt, conversation_instructions_template, None
    )
    expected_instruction = """Please fix the following issue for the repository in /workspace.
An environment has been set up for you to start working. You may assume all necessary tools are installed.

# Problem Statement
Test Issue

This is a test issue ![image](https://sampleimage.com/sample.png)"""

    expected_conversation_instructions = """IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.
You SHOULD INCLUDE PROPER INDENTATION in your edit commands.

When you think you have fixed the issue through code changes, please finish the interaction."""

    assert instruction == expected_instruction
    assert conversation_instructions == expected_conversation_instructions

    assert images_urls == ['https://sampleimage.com/sample.png']


def test_file_instruction_with_repo_instruction():
    issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=123,
        title='Test Issue',
        body='This is a test issue',
    )
    # load prompt from openhands/resolver/prompts/resolve/basic.jinja
    with open('openhands/resolver/prompts/resolve/basic.jinja', 'r') as f:
        prompt = f.read()

    with open(
        'openhands/resolver/prompts/resolve/basic-conversation-instructions.jinja', 'r'
    ) as f:
        conversation_instructions_prompt = f.read()

    # load repo instruction from openhands/resolver/prompts/repo_instructions/all-hands-ai___openhands-resolver.txt
    with open(
        'openhands/resolver/prompts/repo_instructions/all-hands-ai___openhands-resolver.txt',
        'r',
    ) as f:
        repo_instruction = f.read()

    mock_llm_config = LLMConfig(model='test_model', api_key='test_api_key')
    issue_handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), mock_llm_config
    )
    instruction, conversation_instructions, image_urls = issue_handler.get_instruction(
        issue, prompt, conversation_instructions_prompt, repo_instruction
    )

    expected_instruction = """Please fix the following issue for the repository in /workspace.
An environment has been set up for you to start working. You may assume all necessary tools are installed.

# Problem Statement
Test Issue

This is a test issue"""

    expected_conversation_instructions = """IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.
You SHOULD INCLUDE PROPER INDENTATION in your edit commands.

Some basic information about this repository:
This is a Python repo for openhands-resolver, a library that attempts to resolve github issues with the AI agent OpenHands.

- Setup: `poetry install --with test --with dev`
- Testing: `poetry run pytest tests/test_*.py`


When you think you have fixed the issue through code changes, please finish the interaction."""

    assert instruction == expected_instruction
    assert conversation_instructions == expected_conversation_instructions
    assert conversation_instructions is not None
    assert issue_handler.issue_type == 'issue'
    assert image_urls == []


def test_guess_success():
    mock_issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=1,
        title='Test Issue',
        body='This is a test issue',
    )
    mock_history = [create_cmd_output(exit_code=0, content='', command='cd /workspace')]
    mock_llm_config = LLMConfig(model='test_model', api_key='test_api_key')

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [
        MagicMock(
            message=MagicMock(
                content='--- success\ntrue\n--- explanation\nIssue resolved successfully'
            )
        )
    ]
    issue_handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), mock_llm_config
    )

    with patch.object(
        LLM, 'completion', MagicMock(return_value=mock_completion_response)
    ):
        success, comment_success, explanation = issue_handler.guess_success(
            mock_issue, mock_history
        )
        assert issue_handler.issue_type == 'issue'
        assert comment_success is None
        assert success
        assert explanation == 'Issue resolved successfully'


def test_guess_success_with_thread_comments():
    mock_issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=1,
        title='Test Issue',
        body='This is a test issue',
        thread_comments=[
            'First comment',
            'Second comment',
            'latest feedback:\nPlease add tests',
        ],
    )
    mock_history = [MagicMock(message='I have added tests for this case')]
    mock_llm_config = LLMConfig(model='test_model', api_key='test_api_key')

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [
        MagicMock(
            message=MagicMock(
                content='--- success\ntrue\n--- explanation\nTests have been added to verify thread comments handling'
            )
        )
    ]
    issue_handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), mock_llm_config
    )

    with patch.object(
        LLM, 'completion', MagicMock(return_value=mock_completion_response)
    ):
        success, comment_success, explanation = issue_handler.guess_success(
            mock_issue, mock_history
        )
        assert issue_handler.issue_type == 'issue'
        assert comment_success is None
        assert success
        assert 'Tests have been added' in explanation


def test_instruction_with_thread_comments():
    # Create an issue with thread comments
    issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=123,
        title='Test Issue',
        body='This is a test issue',
        thread_comments=[
            'First comment',
            'Second comment',
            'latest feedback:\nPlease add tests',
        ],
    )

    # Load the basic prompt template
    with open('openhands/resolver/prompts/resolve/basic.jinja', 'r') as f:
        prompt = f.read()

    with open(
        'openhands/resolver/prompts/resolve/basic-conversation-instructions.jinja', 'r'
    ) as f:
        conversation_instructions_template = f.read()

    llm_config = LLMConfig(model='test', api_key='test')
    issue_handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), llm_config
    )
    instruction, _, images_urls = issue_handler.get_instruction(
        issue, prompt, conversation_instructions_template, None
    )

    # Verify that thread comments are included in the instruction
    assert 'First comment' in instruction
    assert 'Second comment' in instruction
    assert 'Please add tests' in instruction
    assert 'Issue Thread Comments:' in instruction
    assert images_urls == []


def test_guess_success_failure():
    mock_issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=1,
        title='Test Issue',
        body='This is a test issue',
        thread_comments=[
            'First comment',
            'Second comment',
            'latest feedback:\nPlease add tests',
        ],
    )
    mock_history = [MagicMock(message='I have added tests for this case')]
    mock_llm_config = LLMConfig(model='test_model', api_key='test_api_key')

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [
        MagicMock(
            message=MagicMock(
                content='--- success\ntrue\n--- explanation\nTests have been added to verify thread comments handling'
            )
        )
    ]
    issue_handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), mock_llm_config
    )

    with patch.object(
        LLM, 'completion', MagicMock(return_value=mock_completion_response)
    ):
        success, comment_success, explanation = issue_handler.guess_success(
            mock_issue, mock_history
        )
        assert issue_handler.issue_type == 'issue'
        assert comment_success is None
        assert success
        assert 'Tests have been added' in explanation


def test_guess_success_negative_case():
    mock_issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=1,
        title='Test Issue',
        body='This is a test issue',
    )
    mock_history = [create_cmd_output(exit_code=0, content='', command='cd /workspace')]
    mock_llm_config = LLMConfig(model='test_model', api_key='test_api_key')

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [
        MagicMock(
            message=MagicMock(
                content='--- success\nfalse\n--- explanation\nIssue not resolved'
            )
        )
    ]
    issue_handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), mock_llm_config
    )

    with patch.object(
        LLM, 'completion', MagicMock(return_value=mock_completion_response)
    ):
        success, comment_success, explanation = issue_handler.guess_success(
            mock_issue, mock_history
        )
        assert issue_handler.issue_type == 'issue'
        assert comment_success is None
        assert not success
        assert explanation == 'Issue not resolved'


def test_guess_success_invalid_output():
    mock_issue = Issue(
        owner='test_owner',
        repo='test_repo',
        number=1,
        title='Test Issue',
        body='This is a test issue',
    )
    mock_history = [create_cmd_output(exit_code=0, content='', command='cd /workspace')]
    mock_llm_config = LLMConfig(model='test_model', api_key='test_api_key')

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [
        MagicMock(message=MagicMock(content='This is not a valid output'))
    ]
    issue_handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), mock_llm_config
    )

    with patch.object(
        LLM, 'completion', MagicMock(return_value=mock_completion_response)
    ):
        success, comment_success, explanation = issue_handler.guess_success(
            mock_issue, mock_history
        )
        assert issue_handler.issue_type == 'issue'
        assert comment_success is None
        assert not success
        assert (
            explanation
            == 'Failed to decode answer from LLM response: This is not a valid output'
        )


def test_download_pr_with_review_comments():
    llm_config = LLMConfig(model='test', api_key='test')
    handler = ServiceContextPR(GithubPRHandler('owner', 'repo', 'token'), llm_config)
    mock_pr_response = MagicMock()
    mock_pr_response.json.side_effect = [
        [
            {
                'number': 1,
                'title': 'PR 1',
                'body': 'This is a pull request',
                'head': {'ref': 'b1'},
            },
        ],
        None,
    ]
    mock_pr_response.raise_for_status = MagicMock()

    # Mock for PR comments response
    mock_comments_response = MagicMock()
    mock_comments_response.json.return_value = []  # No PR comments
    mock_comments_response.raise_for_status = MagicMock()

    # Mock for GraphQL request with review comments but no threads
    mock_graphql_response = MagicMock()
    mock_graphql_response.json.side_effect = lambda: {
        'data': {
            'repository': {
                'pullRequest': {
                    'closingIssuesReferences': {'edges': []},
                    'reviews': {
                        'nodes': [
                            {'body': 'Please fix this typo'},
                            {'body': 'Add more tests'},
                        ]
                    },
                }
            }
        }
    }

    mock_graphql_response.raise_for_status = MagicMock()

    def get_mock_response(url, *args, **kwargs):
        if '/comments' in url:
            return mock_comments_response
        return mock_pr_response

    with patch('httpx.get', side_effect=get_mock_response):
        with patch('httpx.post', return_value=mock_graphql_response):
            issues = handler.get_converted_issues(issue_numbers=[1])

    assert len(issues) == 1
    assert handler.issue_type == 'pr'
    assert isinstance(issues[0], Issue)
    assert issues[0].number == 1
    assert issues[0].title == 'PR 1'
    assert issues[0].head_branch == 'b1'

    # Verify review comments are set but threads are empty
    assert len(issues[0].review_comments) == 2
    assert issues[0].review_comments[0] == 'Please fix this typo'
    assert issues[0].review_comments[1] == 'Add more tests'
    assert not issues[0].review_threads
    assert not issues[0].closing_issues
    assert not issues[0].thread_ids


def test_download_issue_with_specific_comment():
    llm_config = LLMConfig(model='test', api_key='test')
    handler = ServiceContextIssue(
        GithubIssueHandler('owner', 'repo', 'token'), llm_config
    )

    # Define the specific comment_id to filter
    specific_comment_id = 101

    # Mock issue and comment responses
    mock_issue_response = MagicMock()
    mock_issue_response.json.side_effect = [
        [
            {'number': 1, 'title': 'Issue 1', 'body': 'This is an issue'},
        ],
        None,
    ]
    mock_issue_response.raise_for_status = MagicMock()

    mock_comments_response = MagicMock()
    mock_comments_response.json.return_value = [
        {
            'id': specific_comment_id,
            'body': 'Specific comment body',
            'issue_url': 'https://api.github.com/repos/owner/repo/issues/1',
        },
        {
            'id': 102,
            'body': 'Another comment body',
            'issue_url': 'https://api.github.com/repos/owner/repo/issues/2',
        },
    ]
    mock_comments_response.raise_for_status = MagicMock()

    def get_mock_response(url, *args, **kwargs):
        if '/comments' in url:
            return mock_comments_response

        return mock_issue_response

    with patch('httpx.get', side_effect=get_mock_response):
        issues = handler.get_converted_issues(
            issue_numbers=[1], comment_id=specific_comment_id
        )

    assert len(issues) == 1
    assert issues[0].number == 1
    assert issues[0].title == 'Issue 1'
    assert issues[0].thread_comments == ['Specific comment body']


if __name__ == '__main__':
    pytest.main()
