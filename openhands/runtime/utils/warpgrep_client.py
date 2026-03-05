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
        context_chars = len(messages[0]['content'])
        last_response = ''

        for turn in range(self.MAX_TURNS):
            response_text = self._call_api(messages)
            if not response_text:
                break
            last_response = response_text

            tool_calls = self._parse_tool_calls(response_text)

            # Check for finish tool call
            for tc in tool_calls:
                if tc['function'] == 'finish':
                    files_param = tc['params'].get('files', '')
                    return self._process_finish(files_param)

            if not tool_calls:
                return []

            messages.append({'role': 'assistant', 'content': response_text})

            tool_results = []
            for tc in tool_calls:
                result = self._execute_tool_call(tc)
                tool_results.append(f'<tool_response>\n{result}\n</tool_response>')

            tool_response_text = '\n\n'.join(tool_results)
            context_chars += len(response_text) + len(tool_response_text)
            remaining = self.MAX_TURNS - turn - 1
            budget_pct = min(100, int(context_chars / 160000 * 100))

            if remaining == 0:
                turn_msg = (
                    f'You have used {turn + 1} turns, you only have 1 turn remaining. '
                    'You have run out of turns to explore the code base and MUST call the finish tool now'
                )
            else:
                turn_msg = f'You have used {turn + 1} turn{"s" if turn + 1 > 1 else ""} and have {remaining} remaining.'

            user_content = (
                f'{tool_response_text}\n\n'
                f'{turn_msg}\n'
                f'<context_budget>{budget_pct}% ({context_chars}/160000 chars)</context_budget>'
            )
            messages.append({'role': 'user', 'content': user_content})

        return []

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
            glob_param = params.get('glob', '')
            glob_flag = f" --glob '{glob_param}'" if glob_param else ''
            cmd = f"rg --line-number --no-heading --color never -C 1{glob_flag} '{safe_pattern}' '{path}' | head -{self.MAX_GREP_LINES}"
            return self.run_in_sandbox(cmd)

        elif func == 'read':
            path = params.get('path', '')
            if not path.startswith('/'):
                path = f'{self.workspace_root}/{path}'
            lines_param = params.get('lines', '')
            if lines_param:
                # Parse line ranges like "1-50" or "1-20,45-80"
                ranges = []
                for part in lines_param.split(','):
                    part = part.strip()
                    if '-' in part:
                        s, e = part.split('-', 1)
                        ranges.append(f'{s.strip()},{e.strip()}p')
                    else:
                        ranges.append(f'{part}p')
                sed_expr = ';'.join(ranges)
                cmd = f"sed -n '{sed_expr}' '{path}' | cat -n"
            else:
                cmd = f"head -{self.MAX_READ_LINES} '{path}' | cat -n"
            return self.run_in_sandbox(cmd)

        elif func == 'list_directory':
            path = params.get('path', self.workspace_root)
            if not path.startswith('/'):
                path = f'{self.workspace_root}/{path}'
            cmd = f"find '{path}' -maxdepth 2 -type f | head -{self.MAX_LIST_LINES}"
            return self.run_in_sandbox(cmd)

        else:
            return f'Unknown tool: {func}'

    def _process_finish(self, files_param: str) -> list[dict]:
        """Process the finish tool call's files parameter.

        Format is newline-delimited file specs:
          path/to/file.py:1-15,45-80
          path/to/other.py:*
          path/to/another.py
        """
        results = []
        if not files_param.strip():
            return results

        for line in files_param.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            if ':' in line:
                path, ranges_str = line.rsplit(':', 1)
                path = path.strip()
            else:
                path = line
                ranges_str = '*'

            if not path.startswith('/'):
                full_path = f'{self.workspace_root}/{path}'
            else:
                full_path = path

            if ranges_str.strip() == '*':
                cmd = f"cat -n '{full_path}'"
                content = self.run_in_sandbox(cmd)
                results.append({
                    'file': path,
                    'start': 1,
                    'end': 0,
                    'content': content,
                })
            else:
                for part in ranges_str.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    if '-' in part:
                        try:
                            start_s, end_s = part.split('-', 1)
                            start = int(start_s.strip())
                            end = int(end_s.strip())
                        except ValueError:
                            continue
                    else:
                        try:
                            start = end = int(part.strip())
                        except ValueError:
                            continue

                    cmd = f"sed -n '{start},{end}p' '{full_path}' | cat -n"
                    content = self.run_in_sandbox(cmd)
                    results.append({
                        'file': path,
                        'start': start,
                        'end': end,
                        'content': content,
                    })

        return results
