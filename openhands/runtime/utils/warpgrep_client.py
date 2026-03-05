"""WarpGrep client that implements the multi-turn WarpGrep protocol.

The client makes API calls from the host process and executes local tool calls
(ripgrep, read, list_dir) inside the sandbox via callbacks. The API key stays
on the host and is never exposed inside the sandbox.
"""

import re
from typing import Callable

import httpx

from openhands.core.logger import openhands_logger as logger


class WarpGrepClient:
    API_URL = 'https://api.morphllm.com/v1/chat/completions'
    MODEL = 'morph-warp-grep-v2'
    MAX_TURNS = 4
    MAX_GREP_LINES = 200
    MAX_READ_LINES = 800
    MAX_LIST_LINES = 200

    def __init__(
        self,
        api_key: str,
        run_in_sandbox_fn: Callable[[str], str],
        workspace_root: str = '/workspace',
    ):
        self.api_key = api_key
        self.run_in_sandbox = run_in_sandbox_fn
        self.workspace_root = workspace_root

    def search(self, query: str) -> list[dict]:
        """Execute a multi-turn WarpGrep search and return results."""
        repo_tree = self._get_repo_tree()
        messages = [
            {
                'role': 'user',
                'content': f'<repo_structure>{repo_tree}</repo_structure>\n<search_string>{query}</search_string>',
            }
        ]

        for turn in range(self.MAX_TURNS):
            response_text = self._call_api(messages)
            if not response_text:
                break

            tool_calls = self._parse_tool_calls(response_text)
            if not tool_calls:
                # Model finished - parse final results
                return self._parse_finish(response_text)

            messages.append({'role': 'assistant', 'content': response_text})

            tool_results = []
            for tc in tool_calls:
                result = self._execute_tool_call(tc)
                tool_results.append(result)

            results_content = '\n'.join(
                f'<tool_response>{r}</tool_response>' for r in tool_results
            )
            results_content += f'\n[Turn {turn + 1}/{self.MAX_TURNS}]'
            messages.append({'role': 'user', 'content': results_content})

        # If we exhausted turns, try to parse whatever we have
        return self._parse_finish(response_text if 'response_text' in dir() else '')

    def _get_repo_tree(self) -> str:
        """Generate a file tree of the workspace."""
        cmd = f'find {self.workspace_root} -type f -not -path "*/.git/*" -not -path "*/node_modules/*" -not -path "*/__pycache__/*" -not -path "*/.venv/*" | head -500'
        return self.run_in_sandbox(cmd)

    def _call_api(self, messages: list[dict]) -> str:
        """Make a completion request to the WarpGrep API."""
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    self.API_URL,
                    headers={
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json',
                    },
                    json={
                        'model': self.MODEL,
                        'messages': messages,
                        'temperature': 0.0,
                        'max_tokens': 2048,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f'WarpGrep API error: {e}')
            return ''

    def _parse_tool_calls(self, text: str) -> list[dict]:
        """Parse XML tool calls from model response."""
        calls = []
        pattern = r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>'
        for match in re.finditer(pattern, text, re.DOTALL):
            func_name = match.group(1)
            params_text = match.group(2)
            params = {}
            param_pattern = r'<parameter=(\w+)>(.*?)</parameter>'
            for pm in re.finditer(param_pattern, params_text, re.DOTALL):
                params[pm.group(1)] = pm.group(2).strip()
            calls.append({'function': func_name, 'params': params})
        return calls

    def _execute_tool_call(self, tool_call: dict) -> str:
        """Execute a tool call in the sandbox and return the result."""
        func = tool_call['function']
        params = tool_call['params']

        if func == 'ripgrep':
            pattern = params.get('pattern', '')
            path = params.get('path', self.workspace_root)
            if not path.startswith('/'):
                path = f'{self.workspace_root}/{path}'
            # Escape single quotes in pattern for shell safety
            safe_pattern = pattern.replace("'", "'\\''")
            cmd = f"rg --line-number --no-heading --color never -C 1 '{safe_pattern}' {path} | head -{self.MAX_GREP_LINES}"
            return self.run_in_sandbox(cmd)

        elif func == 'read':
            path = params.get('path', '')
            if not path.startswith('/'):
                path = f'{self.workspace_root}/{path}'
            start = params.get('start', '1')
            end = params.get('end', str(self.MAX_READ_LINES))
            cmd = f"sed -n '{start},{end}p' {path} | cat -n"
            return self.run_in_sandbox(cmd)

        elif func == 'list_directory':
            path = params.get('path', self.workspace_root)
            if not path.startswith('/'):
                path = f'{self.workspace_root}/{path}'
            cmd = f'find {path} -maxdepth 2 -type f | head -{self.MAX_LIST_LINES}'
            return self.run_in_sandbox(cmd)

        else:
            return f'Unknown tool: {func}'

    def _parse_finish(self, text: str) -> list[dict]:
        """Parse the final results from the model's finish response."""
        results = []
        # Look for file spans in the finish response
        # Pattern: file path followed by line ranges
        span_pattern = r'<file_span>\s*<path>(.*?)</path>\s*<start>(\d+)</start>\s*<end>(\d+)</end>\s*</file_span>'
        for match in re.finditer(span_pattern, text, re.DOTALL):
            path = match.group(1).strip()
            start = int(match.group(2))
            end = int(match.group(3))
            # Read the span from the sandbox
            if not path.startswith('/'):
                path = f'{self.workspace_root}/{path}'
            cmd = f"sed -n '{start},{end}p' {path} | cat -n"
            content = self.run_in_sandbox(cmd)
            results.append(
                {
                    'file': path,
                    'start': start,
                    'end': end,
                    'content': content,
                }
            )

        # If no structured results, return the raw text as a single result
        if not results and text:
            # Strip think tags
            clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            if clean:
                results.append({'file': '', 'start': 0, 'end': 0, 'content': clean})

        return results
