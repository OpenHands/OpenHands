import base64
from typing import Any
from urllib.parse import quote

import httpx

from openhands.core.logger import openhands_logger as logger
from openhands.resolver.interfaces.issue import (
    Issue,
    IssueHandlerInterface,
    ReviewThread,
)
from openhands.resolver.utils import extract_issue_references
from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync
from openhands.utils.http_session import httpx_verify_option


class BitbucketDCIssueHandler(IssueHandlerInterface):
    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str | None = None,
        base_domain: str = 'bitbucket.example.com',
    ):
        """Initialize a Bitbucket Data Center issue handler.

        Args:
            owner: The project key of the repository
            repo: The slug of the repository
            token: The Bitbucket DC API token (user:password or user:token format)
            username: Optional username (used when token is a bare API token)
            base_domain: The hostname of the Bitbucket DC instance
        """
        self.owner = owner
        self.repo = repo
        self.token = token
        self.username = username
        self.base_domain = base_domain
        self.base_url = self.get_base_url()
        self.download_url = self.get_download_url()
        self.clone_url = self.get_clone_url()
        self.headers = self.get_headers()

    def set_owner(self, owner: str) -> None:
        self.owner = owner

    def get_headers(self) -> dict[str, str]:
        # DC always uses HTTP Basic auth
        if ':' in self.token:
            auth_str = base64.b64encode(self.token.encode()).decode()
        elif self.username:
            creds = f'{self.username}:{self.token}'
            auth_str = base64.b64encode(creds.encode()).decode()
        else:
            auth_str = base64.b64encode(self.token.encode()).decode()
        return {
            'Authorization': f'Basic {auth_str}',
            'Accept': 'application/json',
        }

    def get_base_url(self) -> str:
        return f'https://{self.base_domain}/rest/api/1.0'

    def _get_repo_api_base(self) -> str:
        return f'{self.base_url}/projects/{self.owner}/repos/{self.repo}'

    def get_download_url(self) -> str:
        return (
            f'https://{self.base_domain}/rest/api/latest'
            f'/projects/{self.owner}/repos/{self.repo}/archive?format=zip'
        )

    def get_clone_url(self) -> str:
        return f'https://{self.base_domain}/scm/{self.owner.lower()}/{self.repo}.git'

    def get_repo_url(self) -> str:
        return f'https://{self.base_domain}/projects/{self.owner}/repos/{self.repo}'

    def get_issue_url(self, issue_number: int) -> str:
        # DC has no issue tracker; use pull-requests URL
        return f'{self.get_repo_url()}/pull-requests/{issue_number}'

    def get_pr_url(self, pr_number: int) -> str:
        return f'{self.get_repo_url()}/pull-requests/{pr_number}'

    def get_pull_url(self, pr_number: int) -> str:
        return f'{self.get_repo_url()}/pull-requests/{pr_number}'

    def get_branch_url(self, branch_name: str) -> str:
        return f'{self.get_repo_url()}/browse?at=refs/heads/{branch_name}'

    def get_compare_url(self, branch_name: str) -> str:
        default_branch = self.get_default_branch_name()
        return (
            f'{self.get_repo_url()}/compare/commits'
            f'?sourceBranch=refs/heads/{branch_name}'
            f'&targetBranch=refs/heads/{default_branch}'
        )

    def get_authorize_url(self) -> str:
        if ':' in self.token:
            user, _, token = self.token.partition(':')
            creds = f'{quote(user, safe="")}:{quote(token, safe="")}'
        elif self.username:
            creds = f'{quote(self.username, safe="")}:{quote(self.token, safe="")}'
        else:
            creds = quote(self.token, safe='')
        return f'https://{creds}@{self.base_domain}/'

    def get_graphql_url(self) -> str:
        # DC has no GraphQL API; return a placeholder
        return f'https://{self.base_domain}/rest/api/1.0'

    def get_branch_name(self, base_branch_name: str) -> str:
        return f'{base_branch_name}-{self.owner}'

    async def get_issue(self, issue_number: int) -> Issue:
        """Fetch a Bitbucket DC pull request as an Issue.

        Args:
            issue_number: The pull request ID

        Returns:
            An Issue object populated from the DC pull request response
        """
        url = f'{self._get_repo_api_base()}/pull-requests/{issue_number}'
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

        head_branch = data.get('fromRef', {}).get('displayId', '')
        base_branch = data.get('toRef', {}).get('displayId', '')

        return Issue(
            owner=self.owner,
            repo=self.repo,
            number=data.get('id'),
            title=data.get('title', ''),
            body=data.get('description', ''),
            head_branch=head_branch,
            base_branch=base_branch,
        )

    def create_pr(
        self,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> str:
        """Create a pull request on Bitbucket DC.

        Args:
            title: PR title
            body: PR description
            head: Source branch name
            base: Target branch name

        Returns:
            The URL of the created pull request
        """
        result = self.create_pull_request(
            {
                'title': title,
                'description': body,
                'source_branch': head,
                'target_branch': base,
            }
        )
        return result.get('html_url', '')

    def create_pull_request(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a pull request and return html_url and number.

        Args:
            data: Dict with keys title, description, source_branch, target_branch

        Returns:
            Dict with 'html_url' and 'number' keys
        """
        if data is None:
            data = {}

        title = data.get('title', '')
        description = data.get('description', '')
        source_branch = data.get('source_branch', '')
        target_branch = data.get('target_branch', '')

        url = f'{self._get_repo_api_base()}/pull-requests'
        payload = {
            'title': title,
            'description': description,
            'fromRef': {
                'id': f'refs/heads/{source_branch}',
                'repository': {
                    'slug': self.repo,
                    'project': {'key': self.owner},
                },
            },
            'toRef': {
                'id': f'refs/heads/{target_branch}',
                'repository': {
                    'slug': self.repo,
                    'project': {'key': self.owner},
                },
            },
        }
        response = httpx.post(
            url, headers=self.headers, json=payload, verify=httpx_verify_option()
        )
        response.raise_for_status()
        resp_data = response.json()

        links = resp_data.get('links', {}).get('self', [])
        html_url = links[0].get('href', '') if links else ''

        return {
            'html_url': html_url,
            'number': resp_data.get('id', 0),
        }

    def send_comment_msg(self, issue_number: int, msg: str) -> None:
        url = f'{self._get_repo_api_base()}/pull-requests/{issue_number}/comments'
        payload = {'text': msg}
        response = httpx.post(
            url, headers=self.headers, json=payload, verify=httpx_verify_option()
        )
        response.raise_for_status()

    def reply_to_comment(self, pr_number: int, comment_id: str, reply: str) -> None:
        url = f'{self._get_repo_api_base()}/pull-requests/{pr_number}/comments'
        payload = {
            'text': reply,
            'parent': {'id': int(comment_id)},
        }
        response = httpx.post(
            url, headers=self.headers, json=payload, verify=httpx_verify_option()
        )
        response.raise_for_status()

    def branch_exists(self, branch_name: str) -> bool:
        url = f'{self._get_repo_api_base()}/branches'
        params = {'filterText': branch_name, 'limit': 1}
        try:
            response = httpx.get(
                url, headers=self.headers, params=params, verify=httpx_verify_option()
            )
            response.raise_for_status()
            data = response.json()
            values = data.get('values', [])
            return any(v.get('displayId') == branch_name for v in values)
        except httpx.HTTPError as e:
            logger.warning(f'Failed to check branch existence: {e}')
            return False

    def get_default_branch_name(self) -> str:
        url = self._get_repo_api_base()
        try:
            response = httpx.get(
                url, headers=self.headers, verify=httpx_verify_option()
            )
            response.raise_for_status()
            data = response.json()
            default_branch = data.get('defaultBranch', {})
            if default_branch:
                display_id = default_branch.get('displayId', '')
                if display_id:
                    if display_id.startswith('refs/heads/'):
                        return display_id[len('refs/heads/') :]
                    return display_id
        except httpx.HTTPError as e:
            logger.warning(f'Failed to get default branch name: {e}')
        return 'master'

    def download_issues(self) -> list[Any]:
        logger.warning(
            'BitbucketDCIssueHandler.download_issues not implemented; '
            'use get_issue() to fetch individual pull requests'
        )
        return []

    def get_issue_comments(
        self, issue_number: int, comment_id: int | None = None
    ) -> list[str] | None:
        logger.warning('BitbucketDCIssueHandler.get_issue_comments not implemented')
        return []

    def get_issue_thread_comments(self, issue_number: int) -> list[str]:
        logger.warning(
            'BitbucketDCIssueHandler.get_issue_thread_comments not implemented'
        )
        return []

    def get_issue_review_comments(self, issue_number: int) -> list[str]:
        logger.warning(
            'BitbucketDCIssueHandler.get_issue_review_comments not implemented'
        )
        return []

    def get_issue_review_threads(self, issue_number: int) -> list[ReviewThread]:
        logger.warning(
            'BitbucketDCIssueHandler.get_issue_review_threads not implemented'
        )
        return []

    def get_context_from_external_issues_references(
        self,
        closing_issues: list[str],
        closing_issue_numbers: list[int],
        issue_body: str,
        review_comments: list[str] | None,
        review_threads: list[ReviewThread],
        thread_comments: list[str] | None,
    ) -> list[str]:
        # DC has no issue tracker; return closing_issues immediately without API calls
        return closing_issues

    def request_reviewers(self, reviewer: str, pr_number: int) -> None:
        logger.warning('BitbucketDCIssueHandler.request_reviewers not implemented')

    def get_issue_references(self, body: str) -> list[int]:
        return extract_issue_references(body)

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        if not issue_numbers:
            raise ValueError('Unspecified issue numbers')

        converted_issues = []
        for issue_number in issue_numbers:
            try:
                issue = call_async_from_sync(
                    self.get_issue, GENERAL_TIMEOUT, issue_number
                )
                converted_issues.append(issue)
            except Exception as e:
                logger.warning(f'Failed to fetch pull request {issue_number}: {e}')

        return converted_issues


class BitbucketDCPRHandler(BitbucketDCIssueHandler):
    """Handler for Bitbucket Data Center pull requests, extending the issue handler."""

    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str | None = None,
        base_domain: str = 'bitbucket.example.com',
    ):
        super().__init__(owner, repo, token, username, base_domain)
