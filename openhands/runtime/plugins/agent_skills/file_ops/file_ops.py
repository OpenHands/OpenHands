# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
"""File operations module for OpenHands agent.

This module provides a collection of file manipulation skills that enable the OpenHands
agent to perform various file operations such as opening, searching, and navigating
through files and directories.

Functions:
- open_file(path: str, line_number: int | None = 1, context_lines: int = 100): Opens a file and optionally moves to a specific line.
- goto_line(line_number: int): Moves the window to show the specified line number.
- scroll_down(): Moves the window down by the number of lines specified in WINDOW.
- scroll_up(): Moves the window up by the number of lines specified in WINDOW.
- search_dir(search_term: str, dir_path: str = './'): Searches for a term in all files in the specified directory.
- search_file(search_term: str, file_path: str | None = None): Searches for a term in the specified file or the currently open file.
- find_file(file_name: str, dir_path: str = './'): Finds all files with the given name in the specified directory.

Note:
    All functions return string representations of their results.
"""

import os

from openhands.linter import DefaultLinter, LintResult

CURRENT_FILE: str | None = None
CURRENT_LINE = 1
WINDOW = 100

# This is also used in unit tests!
MSG_FILE_UPDATED = '[File updated (edited at line {line_number}). Please review the changes and make sure they are correct (correct indentation, no duplicate lines, etc). Edit the file again if necessary.]'
LINTER_ERROR_MSG = '[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]\n'


# ==================================================================================================


def _output_error(error_msg: str) -> bool:
    print(f'ERROR: {error_msg}')
    return False


def _is_valid_filename(file_name: str) -> bool:
    if not file_name or not isinstance(file_name, str) or not file_name.strip():
        return False
    invalid_chars = '<>:"/\\|?*'
    if os.name == 'nt':  # Windows
        invalid_chars = '<>:"/\\|?*'
    elif os.name == 'posix':  # Unix-like systems
        invalid_chars = '\0'

    for char in invalid_chars:
        if char in file_name:
            return False
    return True


def _is_valid_path(path: str) -> bool:
    if not path or not isinstance(path, str):
        return False
    try:
        return os.path.exists(os.path.normpath(path))
    except PermissionError:
        return False


def _create_paths(file_name: str) -> bool:
    try:
        dirname = os.path.dirname(file_name)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        return True
    except PermissionError:
        return False


def _check_current_file(file_path: str | None = None) -> bool:
    global CURRENT_FILE
    if not file_path:
        file_path = CURRENT_FILE
    if not file_path or not os.path.isfile(file_path):
        return _output_error('No file open. Use the open_file function first.')
    return True


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def _lint_file(file_path: str) -> tuple[str | None, int | None]:
    """Perform linting on a file and identify the first error location.

    Lint the file at the given path and return a tuple with a boolean indicating if there are errors,
    and the line number of the first error, if any.

    Args:
        file_path: str: The path to the file to lint.

    Returns:
    A tuple containing:
        - The lint error message if found, None otherwise
        - The line number of the first error, None if no errors
    """
    linter = DefaultLinter()
    lint_error: list[LintResult] = linter.lint(file_path)
    if not lint_error:
        # Linting successful. No issues found.
        return None, None
    first_error_line = lint_error[0].line if len(lint_error) > 0 else None
    error_text = 'ERRORS:\n' + '\n'.join(
        [f'{file_path}:{err.line}:{err.column}: {err.message}' for err in lint_error]
    )
    return error_text, first_error_line


def _print_window(
    file_path: str | None,
    targeted_line: int,
    window: int,
    return_str: bool = False,
    ignore_window: bool = False,
) -> str:
    global CURRENT_LINE
    if not _check_current_file(file_path) or file_path is None:
        return ''
    with open(file_path) as file:
        content = file.read()

        # Ensure the content ends with a newline character
        if not content.endswith('\n'):
            content += '\n'

        lines = content.splitlines(True)  # Keep all line ending characters
        total_lines = len(lines)

        # cover edge cases
        CURRENT_LINE = _clamp(targeted_line, 1, total_lines)
        half_window = max(1, window // 2)
        if ignore_window:
            # Use CURRENT_LINE as starting line (for e.g. scroll_down)
            start = max(1, CURRENT_LINE)
            end = min(total_lines, CURRENT_LINE + window)
        else:
            # Ensure at least one line above and below the targeted line
            start = max(1, CURRENT_LINE - half_window)
            end = min(total_lines, CURRENT_LINE + half_window)

        # Adjust start and end to ensure at least one line above and below
        if start == 1:
            end = min(total_lines, start + window - 1)
        if end == total_lines:
            start = max(1, end - window + 1)

        output = ''

        # only display this when there's at least one line above
        if start > 1:
            output += f'({start - 1} more lines above)\n'
        else:
            output += '(this is the beginning of the file)\n'
        for i in range(start, end + 1):
            _new_line = f'{i}|{lines[i - 1]}'
            if not _new_line.endswith('\n'):
                _new_line += '\n'
            output += _new_line
        if end < total_lines:
            output += f'({total_lines - end} more lines below)\n'
        else:
            output += '(this is the end of the file)\n'
        output = output.rstrip()

        if return_str:
            return output
        else:
            print(output)
            return ''


def _cur_file_header(current_file: str | None, total_lines: int) -> str:
    if not current_file:
        return ''
    return f'[File: {os.path.abspath(current_file)} ({total_lines} lines total)]\n'


def open_file(
    path: str, line_number: int | None = 1, context_lines: int | None = WINDOW
) -> None:
    """Opens a file in the editor and optionally positions at a specific line.

    The function displays a limited window of content, centered around the specified line
    number if provided. To view the complete file content, the agent should use scroll_down and scroll_up
    commands iteratively.

    Args:
        path: The path to the file to open. Absolute path is recommended.
        line_number: The target line number to center the view on (if possible).
            Defaults to 1.
        context_lines: Maximum number of lines to display in the view window.
            Limited to 100 lines. Defaults to 100.
    """
    global CURRENT_FILE, CURRENT_LINE, WINDOW

    if not os.path.isfile(path):
        _output_error(f'File {path} not found.')
        return

    CURRENT_FILE = os.path.abspath(path)
    with open(CURRENT_FILE) as file:
        total_lines = max(1, sum(1 for _ in file))

    if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
        _output_error(f'Line number must be between 1 and {total_lines}')
        return
    CURRENT_LINE = line_number

    # Override WINDOW with context_lines
    if context_lines is None or context_lines < 1:
        context_lines = WINDOW

    output = _cur_file_header(CURRENT_FILE, total_lines)
    output += _print_window(
        CURRENT_FILE,
        CURRENT_LINE,
        _clamp(context_lines, 1, 100),
        return_str=True,
        ignore_window=False,
    )
    if output.strip().endswith('more lines below)'):
        output += '\n[Use `scroll_down` to view the next 100 lines of the file!]'
    print(output)


def goto_line(line_number: int) -> None:
    """Moves the window to show the specified line number.

    Args:
        line_number: int: The line number to move to.
    """
    global CURRENT_FILE, CURRENT_LINE, WINDOW
    if not _check_current_file():
        return

    with open(str(CURRENT_FILE)) as file:
        total_lines = max(1, sum(1 for _ in file))
    if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
        _output_error(f'Line number must be between 1 and {total_lines}.')
        return

    CURRENT_LINE = _clamp(line_number, 1, total_lines)
    output = _cur_file_header(CURRENT_FILE, total_lines)
    output += _print_window(
        CURRENT_FILE, CURRENT_LINE, WINDOW, return_str=True, ignore_window=False
    )
    print(output)


def scroll_down() -> None:
    """Moves the window down by 100 lines.

    Args:
        None
    """
    global CURRENT_FILE, CURRENT_LINE, WINDOW
    if not _check_current_file():
        return
    with open(str(CURRENT_FILE)) as file:
        total_lines = max(1, sum(1 for _ in file))
    CURRENT_LINE = _clamp(CURRENT_LINE + WINDOW, 1, total_lines)
    output = _cur_file_header(CURRENT_FILE, total_lines)
    output += _print_window(
        CURRENT_FILE, CURRENT_LINE, WINDOW, return_str=True, ignore_window=True
    )
    print(output)


def scroll_up() -> None:
    """Moves the window up by 100 lines.

    Args:
        None
    """
    global CURRENT_FILE, CURRENT_LINE, WINDOW
    if not _check_current_file():
        return
    with open(str(CURRENT_FILE)) as file:
        total_lines = max(1, sum(1 for _ in file))
    CURRENT_LINE = _clamp(CURRENT_LINE - WINDOW, 1, total_lines)
    output = _cur_file_header(CURRENT_FILE, total_lines)
    output += _print_window(
        CURRENT_FILE, CURRENT_LINE, WINDOW, return_str=True, ignore_window=True
    )
    print(output)


class LineNumberError(Exception):
    pass


def search_dir(search_term: str, dir_path: str = './') -> None:
    """Searches for search_term in all files in dir. If dir is not provided, searches in the current directory.

    Args:
        search_term: str: The term to search for.
        dir_path: str: The path to the directory to search.
    """
    if not os.path.isdir(dir_path):
        _output_error(f'Directory {dir_path} not found')
        return
    matches = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.startswith('.'):
                continue
            file_path = os.path.join(root, file)
            with open(file_path, 'r', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if search_term in line:
                        matches.append((file_path, line_num, line.strip()))

    if not matches:
        print(f'No matches found for "{search_term}" in {dir_path}')
        return

    num_matches = len(matches)
    num_files = len(set(match[0] for match in matches))

    if num_files > 100:
        print(
            f'More than {num_files} files matched for "{search_term}" in {dir_path}. Please narrow your search.'
        )
        return

    print(f'[Found {num_matches} matches for "{search_term}" in {dir_path}]')
    for file_path, line_num, line in matches:
        print(f'{file_path} (Line {line_num}): {line}')
    print(f'[End of matches for "{search_term}" in {dir_path}]')


def search_file(search_term: str, file_path: str | None = None) -> None:
    """Searches for search_term in file. If file is not provided, searches in the current open file.

    Args:
        search_term: The term to search for.
        file_path: The path to the file to search.
    """
    global CURRENT_FILE
    if file_path is None:
        file_path = CURRENT_FILE
    if file_path is None:
        _output_error('No file specified or open. Use the open_file function first.')
        return
    if not os.path.isfile(file_path):
        _output_error(f'File {file_path} not found.')
        return

    matches = []
    with open(file_path) as file:
        for i, line in enumerate(file, 1):
            if search_term in line:
                matches.append((i, line.strip()))

    if matches:
        print(f'[Found {len(matches)} matches for "{search_term}" in {file_path}]')
        for match in matches:
            print(f'Line {match[0]}: {match[1]}')
        print(f'[End of matches for "{search_term}" in {file_path}]')
    else:
        print(f'[No matches found for "{search_term}" in {file_path}]')


def find_file(file_name: str, dir_path: str = './') -> None:
    """Finds all files with the given name in the specified directory.

    Args:
        file_name: str: The name of the file to find.
        dir_path: str: The path to the directory to search.
    """
    if not os.path.isdir(dir_path):
        _output_error(f'Directory {dir_path} not found')
        return

    matches = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file_name in file:
                matches.append(os.path.join(root, file))

    if matches:
        print(f'[Found {len(matches)} matches for "{file_name}" in {dir_path}]')
        for match in matches:
            print(f'{match}')
        print(f'[End of matches for "{file_name}" in {dir_path}]')
    else:
        print(f'[No matches found for "{file_name}" in {dir_path}]')


__all__ = [
    'open_file',
    'goto_line',
    'scroll_down',
    'scroll_up',
    'search_dir',
    'search_file',
    'find_file',
]
