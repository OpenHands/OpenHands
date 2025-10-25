"""OpenHands API Python helper for automation tasks.

Default base_url is https://app.all-hands.dev.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import requests


class OpenHandsAPI:
    """Minimal client for interacting with the OpenHands API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = 'https://app.all-hands.dev',
    ):
        """Initialize the API client.

        Args:
            api_key: OpenHands API key. If not provided, will use OPENHANDS_API_KEY env var.
            base_url: Base URL for the OpenHands API. Defaults to https://app.all-hands.dev.
        """
        self.api_key = api_key or os.getenv('OPENHANDS_API_KEY')
        if not self.api_key:
            raise ValueError(
                'API key is required. Set OPENHANDS_API_KEY environment variable or pass api_key parameter.'
            )

        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update(
            {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
        )

    def create_conversation(
        self,
        initial_user_msg: str,
        repository: str | None = None,
        selected_branch: str | None = None,
    ) -> dict[str, Any]:
        """Create a new conversation.

        Args:
            initial_user_msg: The initial message to start the conversation
            repository: Git repository name in format "owner/repo" (optional)
            selected_branch: Git branch to use (optional)

        Returns:
            Response containing conversation_id and status
        """
        conversation_data: dict[str, Any] = {'initial_user_msg': initial_user_msg}
        if repository:
            conversation_data['repository'] = repository
        if selected_branch:
            conversation_data['selected_branch'] = selected_branch

        response = self.session.post(
            f'{self.base_url}/api/conversations', json=conversation_data
        )
        response.raise_for_status()
        return response.json()

    def create_conversation_from_files(
        self,
        main_prompt_path: str,
        repository: str | None = None,
        append_common_tail: bool = True,
        common_tail_path: str = 'scripts/prompts/common_tail.j2',
    ) -> dict[str, Any]:
        """Create a conversation by reading a prompt file and optional common tail."""
        main_text = Path(main_prompt_path).read_text()
        if append_common_tail and Path(common_tail_path).exists():
            tail = Path(common_tail_path).read_text()
            initial_user_msg = f'{main_text}\n\n{tail}'
        else:
            initial_user_msg = main_text
        return self.create_conversation(
            initial_user_msg=initial_user_msg,
            repository=repository,
        )

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Get conversation status and details."""
        response = self.session.get(
            f'{self.base_url}/api/conversations/{conversation_id}'
        )
        response.raise_for_status()
        return response.json()

    def get_trajectory(self, conversation_id: str) -> dict[str, Any]:
        """Get the trajectory (event history) for a conversation."""
        response = self.session.get(
            f'{self.base_url}/api/conversations/{conversation_id}/trajectory'
        )
        response.raise_for_status()
        return response.json()

    def poll_until_stopped(
        self, conversation_id: str, timeout: int = 1200, poll_interval: int = 30
    ) -> dict[str, Any]:
        """Poll conversation until it stops or times out."""
        start = time.time()
        while time.time() - start < timeout:
            convo = self.get_conversation(conversation_id)
            status = str(convo.get('status', '')).upper()
            if status == 'STOPPED':
                return convo
            if status in ['FAILED', 'ERROR', 'CANCELLED']:
                return convo
            time.sleep(poll_interval)
        raise TimeoutError(
            f'Conversation {conversation_id} did not stop within {timeout} seconds'
        )

    def post_github_comment(
        self, repo: str, issue_number: int, comment: str, token: str
    ) -> None:
        """Post a comment to a GitHub issue/PR."""
        url = f'https://api.github.com/repos/{repo}/issues/{issue_number}/comments'
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
        }
        data = {'body': comment}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()


def cli_create_convo_from_prompt():
    parser = argparse.ArgumentParser(description='Create an OpenHands conversation from a prompt file')
    parser.add_argument('--prompt', required=True, help='Path to the main prompt file (e.g., scripts/prompts/architecture_refresh.j2)')
    parser.add_argument('--repo', required=False, help='owner/repo to associate in conversation')
    parser.add_argument('--append-common-tail', action='store_true', help='Append common_tail.j2 if present')
    parser.add_argument('--common-tail', default='scripts/prompts/common_tail.j2', help='Path to common tail file')
    parser.add_argument('--base-url', default='https://app.all-hands.dev', help='OpenHands API base URL')
    args = parser.parse_args()

    client = OpenHandsAPI(base_url=args.base_url)
    resp = client.create_conversation_from_files(
        main_prompt_path=args.prompt,
        repository=args.repo,
        append_common_tail=args.append_common_tail,
        common_tail_path=args.common_tail,
    )
    print(resp.get('conversation_id') or resp)


if __name__ == '__main__':
    cli_create_convo_from_prompt()

