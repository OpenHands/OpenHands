import os
import re
import tempfile
from abc import ABC, abstractmethod
from typing import Any

from openhands_aci.utils.diff import get_diff  # type: ignore

from openhands.core.config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from openhands.events.observation import (
    ErrorObservation,
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
    Observation,
)
from openhands.linter import DefaultLinter
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.utils.chunk_localizer import Chunk, get_top_k_chunk_matches

USER_MSG = """
Code changes will be provided in the form of a draft. You will need to apply the draft to the original code.
The original code will be enclosed within `<original_code>` tags.
The draft will be enclosed within `<update_snippet>` tags.
You need to output the update code within `<updated_code>` tags.

Within the `<updated_code>` tag, include only the final code after updation. Do not include any explanations or other content within these tags.

<original_code>{old_contents}</original_code>

<update_snippet>{draft_changes}</update_snippet>
    """

CORRECT_SYS_MSG = """You are a code repair assistant. Now you have an original file content and error information from a static code checking tool (lint tool). Your task is to automatically modify and return the repaired complete code based on these error messages and refer to the current file content.

The following are the specific task steps you need to complete:

Carefully read the current file content to ensure that you fully understand its code structure.

According to the lint error prompt, accurately locate and analyze the cause of the problem.

Modify the original file content and fix all errors prompted by the lint tool.

Return complete, runnable, and error-fixed code, paying attention to maintaining the overall style and specifications of the original code.

Please note:

Please strictly follow the lint error prompts to make modifications and do not miss any problems.

The modified code must be complete and cannot introduce new errors or bugs.

The modified code must maintain the original code function and logic, and no changes unrelated to error repair should be made."""

CORRECT_USER_MSG = """
THE FOLLOWING ARE THE ORIGINAL FILE CONTENTS AND THE ERROR INFORMATION REPORTED BY THE LINT TOOL

# CURRENT FILE CONTENT:
```
{file_content}
```

# ERROR MESSAGE FROM STATIC CODE CHECKING TOOL:
```
{lint_error}
```
""".strip()


def _extract_code(string: str) -> str | None:
    pattern = r'<updated_code>(.*?)</updated_code>'
    matches = re.findall(pattern, string, re.DOTALL)
    if not matches:
        return None

    content = str(matches[0])
    if content.startswith('#EDIT:'):
        # Remove first line
        content = content[content.find('\n') + 1 :]
    return content


def get_new_file_contents(
    llm: LLM, old_contents: str, draft_changes: str, num_retries: int = 3
) -> str | None:
    while num_retries > 0:
        messages = [
            {
                'role': 'user',
                'content': USER_MSG.format(
                    old_contents=old_contents, draft_changes=draft_changes
                ),
            },
        ]
        resp = llm.completion(messages=messages)
        new_contents = _extract_code(resp['choices'][0]['message']['content'])
        if new_contents is not None:
            return new_contents
        num_retries -= 1
    return None


class FileEditRuntimeInterface(ABC):
    config: OpenHandsConfig

    @abstractmethod
    def read(self, action: FileReadAction) -> Observation:
        pass

    @abstractmethod
    def write(self, action: FileWriteAction) -> Observation:
        pass

    @abstractmethod
    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        pass


class FileEditRuntimeMixin(FileEditRuntimeInterface):
    # Most LLMs have output token limit of 4k tokens.
    # This restricts the number of lines we can edit to avoid exceeding the token limit.
    MAX_LINES_TO_EDIT = 300

    def __init__(
        self,
        enable_llm_editor: bool,
        llm_registry: LLMRegistry,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_llm_editor = enable_llm_editor

        if not self.enable_llm_editor:
            return

        draft_editor_config = self.config.get_llm_config('draft_editor')

        # manually set the model name for the draft editor LLM to distinguish token costs
        if draft_editor_config.caching_prompt:
            logger.debug(
                'It is not recommended to cache draft editor LLM prompts as it may incur high costs for the same prompt. '
                'Automatically setting caching_prompt=false.'
            )
            draft_editor_config.caching_prompt = False

        self.draft_editor_llm = llm_registry.get_llm(
            'draft_editor_llm', draft_editor_config
        )
        logger.debug(
            f'[Draft edit functionality] enabled with LLM: {self.draft_editor_llm}'
        )

    def _validate_range(
        self, start: int, end: int, total_lines: int
    ) -> Observation | None:
        # start and end are 1-indexed and inclusive
        if (
            (start < 1 and start != -1)
            or start > total_lines
            or (start > end and end != -1 and start != -1)
        ):
            return ErrorObservation(
                f'Invalid range for editing: start={start}, end={end}, total lines={total_lines}. start must be >= 1 and <={total_lines} (total lines of the edited file), start <= end, or start == -1 (append to the end of the file).'
            )
        if (
            (end < 1 and end != -1)
            or end > total_lines
            or (end < start and start != -1 and end != -1)
        ):
            return ErrorObservation(
                f'Invalid range for editing: start={start}, end={end}, total lines={total_lines}. end must be >= 1 and <= {total_lines} (total lines of the edited file), end >= start, or end == -1 (to edit till the end of the file).'
            )
        return None

    def _get_lint_error(
        self,
        suffix: str,
        old_content: str,
        new_content: str,
        filepath: str,
        diff: str,
    ) -> ErrorObservation | None:
        linter = DefaultLinter()
        # Copy the original file to a temporary file (with the same ext) and lint it
        with (
            tempfile.NamedTemporaryFile(
                suffix=suffix, mode='w+', encoding='utf-8'
            ) as original_file_copy,
            tempfile.NamedTemporaryFile(
                suffix=suffix, mode='w+', encoding='utf-8'
            ) as updated_file_copy,
        ):
            # Lint the original file
            original_file_copy.write(old_content)
            original_file_copy.flush()

            # Lint the updated file
            updated_file_copy.write(new_content)
            updated_file_copy.flush()

            updated_lint_error = linter.lint_file_diff(
                original_file_copy.name, updated_file_copy.name
            )

            if len(updated_lint_error) > 0:
                _obs = FileEditObservation(
                    content=diff,
                    path=filepath,
                    prev_exist=True,
                    old_content=old_content,
                    new_content=new_content,
                )
                error_message = (
                    (
                        f'\n[Linting failed for edited file {filepath}. {len(updated_lint_error)} lint errors found.]\n'
                        '[begin attempted changes]\n'
                        f'{_obs.visualize_diff(change_applied=False)}\n'
                        '[end attempted changes]\n'
                    )
                    + '-' * 40
                    + '\n'
                )
                error_message += '-' * 20 + 'First 5 lint errors' + '-' * 20 + '\n'
                for i, lint_error in enumerate(updated_lint_error[:5]):
                    error_message += f'[begin lint error {i}]\n'
                    error_message += lint_error.visualize().strip() + '\n'
                    error_message += f'[end lint error {i}]\n'
                    error_message += '-' * 40 + '\n'
                return ErrorObservation(error_message)
        return None

    def llm_based_edit(self, action: FileEditAction, retry_num: int = 0) -> Observation:
        obs = self.read(FileReadAction(path=action.path))
        if (
            isinstance(obs, ErrorObservation)
            and 'File not found'.lower() in obs.content.lower()
        ):
            logger.debug(
                f'Agent attempted to edit a file that does not exist. Creating the file. Error msg: {obs.content}'
            )
            # directly write the new content
            obs = self.write(
                FileWriteAction(path=action.path, content=action.content.strip())
            )
            if isinstance(obs, ErrorObservation):
                return obs
            if not isinstance(obs, FileWriteObservation):
                raise ValueError(
                    f'Expected FileWriteObservation, got {type(obs)}: {str(obs)}'
                )
            return FileEditObservation(
                content=get_diff('', action.content, action.path),
                path=action.path,
                prev_exist=False,
                old_content='',
                new_content=action.content,
            )
        if not isinstance(obs, FileReadObservation):
            raise ValueError(
                f'Expected FileReadObservation, got {type(obs)}: {str(obs)}'
            )

        original_file_content = obs.content
        old_file_lines = original_file_content.split('\n')
        # NOTE: start and end are 1-indexed
        start = action.start
        end = action.end
        # validate the range
        error = self._validate_range(start, end, len(old_file_lines))
        if error is not None:
            return error

        # append to the end of the file
        if start == -1:
            updated_content = '\n'.join(old_file_lines + action.content.split('\n'))
            diff = get_diff(original_file_content, updated_content, action.path)
            # Lint the updated content
            if self.config.sandbox.enable_auto_lint:
                suffix = os.path.splitext(action.path)[1]

                error_obs = self._get_lint_error(
                    suffix,
                    original_file_content,
                    updated_content,
                    action.path,
                    diff,
                )
                if error_obs is not None:
                    self.write(
                        FileWriteAction(path=action.path, content=updated_content)
                    )
                    return self.correct_edit(
                        file_content=updated_content,
                        error_obs=error_obs,
                        retry_num=retry_num,
                    )

            obs = self.write(FileWriteAction(path=action.path, content=updated_content))
            return FileEditObservation(
                content=diff,
                path=action.path,
                prev_exist=True,
                old_content=original_file_content,
                new_content=updated_content,
            )

        # Get the 0-indexed start and end
        start_idx = start - 1
        if end != -1:
            # remove 1 to make it 0-indexed
            # then add 1 since the `end` is inclusive
            end_idx = end - 1 + 1
        else:
            # end == -1 means the user wants to edit till the end of the file
            end_idx = len(old_file_lines)

        # Get the range of lines to edit - reject if too long
        length_of_range = end_idx - start_idx
        if length_of_range > self.MAX_LINES_TO_EDIT + 1:
            error_msg = (
                f'[Edit error: The range of lines to edit is too long.]\n'
                f'[The maximum number of lines allowed to edit at once is {self.MAX_LINES_TO_EDIT}. '
                f'Got (L{start_idx + 1}-L{end_idx}) {length_of_range} lines.]\n'
                # [start_idx, end_idx), so no need to + 1
            )
            # search for relevant ranges to hint the agent
            topk_chunks: list[Chunk] = get_top_k_chunk_matches(
                text=original_file_content,
                query=action.content,  # edit draft as query
                k=3,
                max_chunk_size=20,  # lines
            )
            error_msg += (
                'Here are some snippets that maybe relevant to the provided edit.\n'
            )
            for i, chunk in enumerate(topk_chunks):
                error_msg += f'[begin relevant snippet {i + 1}. Line range: L{chunk.line_range[0]}-L{chunk.line_range[1]}. Similarity: {chunk.normalized_lcs}]\n'
                error_msg += f'[Browse around it via `open_file("{action.path}", {(chunk.line_range[0] + chunk.line_range[1]) // 2})`]\n'
                error_msg += chunk.visualize() + '\n'
                error_msg += f'[end relevant snippet {i + 1}]\n'
                error_msg += '-' * 40 + '\n'

            error_msg += 'Consider using `open_file` to explore around the relevant snippets if needed.\n'
            error_msg += f'**IMPORTANT**: Please REDUCE the range of edits to less than {self.MAX_LINES_TO_EDIT} lines by setting `start` and `end` in the edit action (e.g. `<file_edit path="{action.path}" start=[PUT LINE NUMBER HERE] end=[PUT LINE NUMBER HERE] />`). '

            return ErrorObservation(error_msg)

        content_to_edit = '\n'.join(old_file_lines[start_idx:end_idx])
        _edited_content = get_new_file_contents(
            self.draft_editor_llm, content_to_edit, action.content
        )
        if _edited_content is None:
            ret_err = ErrorObservation(
                'Failed to get new file contents. '
                'Please try to reduce the number of edits and try again.'
            )
            ret_err.llm_metrics = self.draft_editor_llm.metrics
            return ret_err

        # piece the updated content with the unchanged content
        updated_lines = (
            old_file_lines[:start_idx]
            + _edited_content.split('\n')
            + old_file_lines[end_idx:]
        )
        updated_content = '\n'.join(updated_lines)
        diff = get_diff(original_file_content, updated_content, action.path)

        # Lint the updated content
        if self.config.sandbox.enable_auto_lint:
            suffix = os.path.splitext(action.path)[1]
            error_obs = self._get_lint_error(
                suffix, original_file_content, updated_content, action.path, diff
            )
            if error_obs is not None:
                error_obs.llm_metrics = self.draft_editor_llm.metrics
                self.write(FileWriteAction(path=action.path, content=updated_content))
                return self.correct_edit(
                    file_content=updated_content,
                    error_obs=error_obs,
                    retry_num=retry_num,
                )

        obs = self.write(FileWriteAction(path=action.path, content=updated_content))
        ret_obs = FileEditObservation(
            content=diff,
            path=action.path,
            prev_exist=True,
            old_content=original_file_content,
            new_content=updated_content,
        )
        ret_obs.llm_metrics = self.draft_editor_llm.metrics
        return ret_obs

    def check_retry_num(self, retry_num: int) -> bool:
        correct_num = self.draft_editor_llm.config.correct_num
        return correct_num < retry_num

    def correct_edit(
        self, file_content: str, error_obs: ErrorObservation, retry_num: int = 0
    ) -> Observation:
        import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
        from openhands.agenthub.codeact_agent.tools import LLMBasedFileEditTool
        from openhands.llm.llm_utils import check_tools

        _retry_num = retry_num + 1
        if self.check_retry_num(_retry_num):
            return error_obs
        tools = check_tools([LLMBasedFileEditTool], self.draft_editor_llm.config)
        messages = [
            {'role': 'system', 'content': CORRECT_SYS_MSG},
            {
                'role': 'user',
                'content': CORRECT_USER_MSG.format(
                    file_content=file_content, lint_error=error_obs.content
                ),
            },
        ]
        params: dict = {'messages': messages, 'tools': tools}
        try:
            response = self.draft_editor_llm.completion(**params)
            actions = codeact_function_calling.response_to_actions(response)
            if len(actions) != 1:
                return error_obs
            for action in actions:
                if isinstance(action, FileEditAction):
                    return self.llm_based_edit(action, _retry_num)
        except Exception as e:
            logger.error(f'correct lint error is failed: {e}')
        return error_obs
