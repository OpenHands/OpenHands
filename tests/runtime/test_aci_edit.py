"""Editor-related tests for the DockerRuntime."""

import os
from unittest.mock import MagicMock

from conftest import _close_test_runtime, _load_runtime

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import FileEditAction, FileWriteAction
from openhands.runtime.action_execution_server import _execute_file_editor
from openhands.runtime.impl.cli.cli_runtime import CLIRuntime


def test_view_file(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Create test file
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='This is a test file.\nThis file is for testing purposes.',
            path=test_file,
        )
        obs = runtime.run_action(action)

        # Test view command
        action = FileEditAction(
            command='view',
            path=test_file,
        )
        obs = runtime.run_action(action)

        assert f"Here's the result of running `cat -n` on {test_file}:" in obs.content
        assert '1\tThis is a test file.' in obs.content
        assert '2\tThis file is for testing purposes.' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_view_directory(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(
        temp_dir, runtime_cls, run_as_openhands, enable_browser=True
    )
    try:
        # Create test file
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='This is a test file.\nThis file is for testing purposes.',
            path=test_file,
        )
        obs = runtime.run_action(action)

        # Test view command
        action = FileEditAction(
            command='view',
            path=config.workspace_mount_path_in_sandbox,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            obs.content
            == f"""Here's the files and directories up to 2 levels deep in {config.workspace_mount_path_in_sandbox}, excluding hidden items:
{config.workspace_mount_path_in_sandbox}/
{config.workspace_mount_path_in_sandbox}/test.txt

1 hidden files/directories in this directory are excluded. You can use 'ls -la /workspace' to see them."""  # The hidden dir is the /workspace/.downloads
        )

    finally:
        _close_test_runtime(runtime)


def test_create_file(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        new_file = os.path.join(config.workspace_mount_path_in_sandbox, 'new_file.txt')
        action = FileEditAction(
            command='create',
            path=new_file,
            file_text='New file content',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'File created successfully' in obs.content

        # Verify file content
        action = FileEditAction(
            command='view',
            path=new_file,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'New file content' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_create_file_with_empty_content(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        new_file = os.path.join(config.workspace_mount_path_in_sandbox, 'new_file.txt')
        action = FileEditAction(
            command='create',
            path=new_file,
            file_text='',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'File created successfully' in obs.content

        # Verify file content
        action = FileEditAction(
            command='view',
            path=new_file,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert '1\t' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_create_with_none_file_text(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        new_file = os.path.join(
            config.workspace_mount_path_in_sandbox, 'none_content.txt'
        )
        action = FileEditAction(
            command='create',
            path=new_file,
            file_text=None,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            obs.content
            == 'ERROR:\nParameter `file_text` is required for command: create.'
        )
    finally:
        _close_test_runtime(runtime)


def test_str_replace(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Create test file
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='This is a test file.\nThis file is for testing purposes.',
            path=test_file,
        )
        runtime.run_action(action)

        # Test str_replace command
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str='test file',
            new_str='sample file',
        )
        obs = runtime.run_action(action)
        assert f'The file {test_file} has been edited' in obs.content

        # Verify file content
        action = FileEditAction(
            command='view',
            path=test_file,
        )
        obs = runtime.run_action(action)
        assert 'This is a sample file.' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_str_replace_multi_line(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='This is a test file.\nThis file is for testing purposes.',
            path=test_file,
        )
        runtime.run_action(action)

        # Test str_replace command
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str='This is a test file.\nThis file is for testing purposes.',
            new_str='This is a sample file.\nThis file is for testing purposes.',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert f'The file {test_file} has been edited.' in obs.content
        assert 'This is a sample file.' in obs.content
        assert 'This file is for testing purposes.' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_str_replace_multi_line_with_tabs(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileEditAction(
            command='create',
            path=test_file,
            file_text='def test():\n\tprint("Hello, World!")',
        )
        runtime.run_action(action)

        # Test str_replace command
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str='def test():\n\tprint("Hello, World!")',
            new_str='def test():\n\tprint("Hello, Universe!")',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            obs.content
            == f"""The file {test_file} has been edited. Here's the result of running `cat -n` on a snippet of {test_file}:
     1\tdef test():
     2\t\tprint("Hello, Universe!")
Review the changes and make sure they are as expected. Edit the file again if necessary."""
        )

    finally:
        _close_test_runtime(runtime)


def test_str_replace_error_multiple_occurrences(
    temp_dir, runtime_cls, run_as_openhands
):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='This is a test file.\nThis file is for testing purposes.',
            path=test_file,
        )
        runtime.run_action(action)

        action = FileEditAction(
            command='str_replace', path=test_file, old_str='test', new_str='sample'
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Multiple occurrences of old_str `test`' in obs.content
        assert '[1, 2]' in obs.content  # Should show both line numbers
    finally:
        _close_test_runtime(runtime)


def test_str_replace_error_multiple_multiline_occurrences(
    temp_dir, runtime_cls, run_as_openhands
):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        # Create a file with two identical multi-line blocks
        multi_block = """def example():
        print("Hello")
        return True"""
        content = f"{multi_block}\n\nprint('separator')\n\n{multi_block}"
        action = FileWriteAction(
            content=content,
            path=test_file,
        )
        runtime.run_action(action)

        # Test str_replace command
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str=multi_block,
            new_str='def new():\n    print("World")',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Multiple occurrences of old_str' in obs.content
        assert '[1, 7]' in obs.content  # Should show correct starting line numbers

    finally:
        _close_test_runtime(runtime)


def test_str_replace_nonexistent_string(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2',
            path=test_file,
        )
        runtime.run_action(action)
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str='Non-existent Line',
            new_str='New Line',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'No replacement was performed' in obs.content
        assert (
            f'old_str `Non-existent Line` did not appear verbatim in {test_file}'
            in obs.content
        )
    finally:
        _close_test_runtime(runtime)


def test_str_replace_with_empty_new_str(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine to remove\nLine 3',
            path=test_file,
        )
        runtime.run_action(action)
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str='Line to remove\n',
            new_str='',
        )
        obs = runtime.run_action(action)
        assert 'Line to remove' not in obs.content
        assert 'Line 1' in obs.content
        assert 'Line 3' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_str_replace_with_empty_old_str(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2\nLine 3',
            path=test_file,
        )
        runtime.run_action(action)
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str='',
            new_str='New string',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if isinstance(runtime, CLIRuntime):
            # CLIRuntime with a 3-line file without a trailing newline reports 3 occurrences for an empty old_str
            assert (
                'No replacement was performed. Multiple occurrences of old_str `` in lines [1, 2, 3]. Please ensure it is unique.'
                in obs.content
            )
        else:
            # Other runtimes might behave differently (e.g., implicitly add a newline, leading to 4 matches)
            # TODO: Why do they have 4 lines?
            assert (
                'No replacement was performed. Multiple occurrences of old_str `` in lines [1, 2, 3, 4]. Please ensure it is unique.'
                in obs.content
            )
    finally:
        _close_test_runtime(runtime)


def test_str_replace_with_none_old_str(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2\nLine 3',
            path=test_file,
        )
        runtime.run_action(action)

        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str=None,
            new_str='new content',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'old_str' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_insert(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Create test file
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2',
            path=test_file,
        )
        runtime.run_action(action)

        # Test insert command
        action = FileEditAction(
            command='insert',
            path=test_file,
            insert_line=1,
            new_str='Inserted line',
        )
        obs = runtime.run_action(action)
        assert f'The file {test_file} has been edited' in obs.content

        # Verify file content
        action = FileEditAction(
            command='view',
            path=test_file,
        )
        obs = runtime.run_action(action)
        assert 'Line 1' in obs.content
        assert 'Inserted line' in obs.content
        assert 'Line 2' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_insert_invalid_line(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2',
            path=test_file,
        )
        runtime.run_action(action)
        action = FileEditAction(
            command='insert',
            path=test_file,
            insert_line=10,
            new_str='Invalid Insert',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Invalid `insert_line` parameter' in obs.content
        assert 'It should be within the range of allowed values' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_insert_with_empty_string(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2',
            path=test_file,
        )
        runtime.run_action(action)
        action = FileEditAction(
            command='insert',
            path=test_file,
            insert_line=1,
            new_str='',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert '1\tLine 1' in obs.content
        assert '2\t\n' in obs.content
        assert '3\tLine 2' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_insert_with_none_new_str(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2',
            path=test_file,
        )
        runtime.run_action(action)

        action = FileEditAction(
            command='insert',
            path=test_file,
            insert_line=1,
            new_str=None,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'ERROR' in obs.content
        assert 'Parameter `new_str` is required for command: insert' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_undo_edit(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Create test file
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='This is a test file.',
            path=test_file,
        )
        runtime.run_action(action)

        # Make an edit
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str='test',
            new_str='sample',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'This is a sample file.' in obs.content

        # Undo the edit
        action = FileEditAction(
            command='undo_edit',
            path=test_file,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Last edit to' in obs.content
        assert 'This is a test file.' in obs.content

        # Verify file content
        action = FileEditAction(
            command='view',
            path=test_file,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'This is a test file.' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_validate_path_invalid(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        invalid_file = os.path.join(
            config.workspace_mount_path_in_sandbox, 'nonexistent.txt'
        )
        action = FileEditAction(
            command='view',
            path=invalid_file,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Invalid `path` parameter' in obs.content
        assert f'The path {invalid_file} does not exist' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_create_existing_file_error(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2',
            path=test_file,
        )
        runtime.run_action(action)
        action = FileEditAction(
            command='create',
            path=test_file,
            file_text='New content',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'File already exists' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_str_replace_missing_old_str(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2',
            path=test_file,
        )
        runtime.run_action(action)
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str='',
            new_str='sample',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            'No replacement was performed. Multiple occurrences of old_str ``'
            in obs.content
        )
    finally:
        _close_test_runtime(runtime)


def test_str_replace_new_str_and_old_str_same(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2',
            path=test_file,
        )
        runtime.run_action(action)
        action = FileEditAction(
            command='str_replace',
            path=test_file,
            old_str='test file',
            new_str='test file',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            'No replacement was performed. `new_str` and `old_str` must be different.'
            in obs.content
        )
    finally:
        _close_test_runtime(runtime)


def test_insert_missing_line_param(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        test_file = os.path.join(config.workspace_mount_path_in_sandbox, 'test.txt')
        action = FileWriteAction(
            content='Line 1\nLine 2',
            path=test_file,
        )
        runtime.run_action(action)
        action = FileEditAction(
            command='insert',
            path=test_file,
            new_str='Missing insert line',
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Parameter `insert_line` is required for command: insert' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_undo_edit_no_history_error(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        empty_file = os.path.join(config.workspace_mount_path_in_sandbox, 'empty.txt')
        action = FileWriteAction(
            content='',
            path=empty_file,
        )
        runtime.run_action(action)

        action = FileEditAction(
            command='undo_edit',
            path=empty_file,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'No edit history found for' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_view_large_file_with_truncation(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Create a large file to trigger truncation
        large_file = os.path.join(
            config.workspace_mount_path_in_sandbox, 'large_test.txt'
        )
        large_content = 'Line 1\n' * 16000  # 16000 lines should trigger truncation
        action = FileWriteAction(
            content=large_content,
            path=large_file,
        )
        runtime.run_action(action)

        action = FileEditAction(
            command='view',
            path=large_file,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            'Due to the max output limit, only part of this file has been shown to you.'
            in obs.content
        )
    finally:
        _close_test_runtime(runtime)


def test_insert_line_string_conversion():
    """Test that insert_line is properly converted from string to int.

    This test reproduces issue #8369 Example 2 where a string value for insert_line
    causes a TypeError in the editor.
    """
    # Mock the OHEditor
    mock_editor = MagicMock()
    mock_editor.return_value = MagicMock(
        error=None, output='Success', old_content=None, new_content=None
    )

    # Test with string insert_line
    result, _ = _execute_file_editor(
        editor=mock_editor,
        command='insert',
        path='/test/path.py',
        insert_line='185',  # String instead of int
        new_str='test content',
    )

    # Verify the editor was called with the correct parameters (insert_line converted to int)
    mock_editor.assert_called_once()
    args, kwargs = mock_editor.call_args
    assert isinstance(kwargs['insert_line'], int)
    assert kwargs['insert_line'] == 185
    assert result == 'Success'
