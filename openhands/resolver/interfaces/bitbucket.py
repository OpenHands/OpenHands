import base64
from typing import Any

import httpx

from openhands.core.logger import openhands_logger as logger
from openhands.resolver.interfaces.issue import (
    Issue,
    IssueHandlerInterface,
    ReviewThread,
)
from openhands.resolver.utils import extract_issue_references
from openhands.utils.http_session import httpx_verify_option


class BitbucketIssueHandler(IssueHandlerInterface):
    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str | None = None,
        base_domain: str = 'bitbucket.org',
    ):
        """Initialize a Bitbucket issue handler.

        Args:
            owner: The workspace of the repository
            repo: The name of the repository
            token: The Bitbucket API token
            username: Optional Bitbucket username
            base_domain: The domain for Bitbucket Server (default: "bitbucket.org")
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
        # Check if the token contains a colon, which indicates it's in username:password format
        if ':' in self.token:
            auth_str = base64.b64encode(self.token.encode()).decode()
            return {
                'Authorization': f'Basic {auth_str}',
                'Accept': 'application/json',
            }
        else:
            return {
                'Authorization': f'Bearer {self.token}',
                'Accept': 'application/json',
            }

    def get_base_url(self) -> str:
        """Get the base URL for the Bitbucket API."""
        return f'https://api.{self.base_domain}/2.0'

    def get_download_url(self) -> str:
        """Get the download URL for the repository."""
        return f'https://{self.base_domain}/{self.owner}/{self.repo}/get/master.zip'

    def get_clone_url(self) -> str:
        """Get the clone URL for the repository."""
        return f'https://{self.base_domain}/{self.owner}/{self.repo}.git'

    def get_repo_url(self) -> str:
        """Get the URL for the repository."""
        return f'https://{self.base_domain}/{self.owner}/{self.repo}'

    def get_issue_url(self, issue_number: int) -> str:
        """Get the URL for an issue."""
        return f'{self.get_repo_url()}/issues/{issue_number}'

    def get_pr_url(self, pr_number: int) -> str:
        """Get the URL for a pull request."""
        return f'{self.get_repo_url()}/pull-requests/{pr_number}'

    async def get_issue(self, issue_number: int) -> Issue:
        """Get an issue from Bitbucket.

        Args:
            issue_number: The issue number

        Returns:
            An Issue object
        """
        url = f'{self.base_url}/repositories/{self.owner}/{self.repo}/issues/{issue_number}'
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

        # Create a basic Issue object with required fields
        issue = Issue(
            owner=self.owner,
            repo=self.repo,
            number=data.get('id'),
            title=data.get('title', ''),
            body=data.get('content', {}).get('raw', ''),
        )

        return issue

    def create_pr(
        self,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> str:
        """Create a pull request.

        Args:
            title: The title of the pull request
            body: The body of the pull request
            head: The head branch
            base: The base branch

        Returns:
            The URL of the created pull request
        """
        url = f'{self.base_url}/repositories/{self.owner}/{self.repo}/pullrequests'

        payload = {
            'title': title,
            'description': body,
            'source': {'branch': {'name': head}},
            'destination': {'branch': {'name': base}},
            'close_source_branch': False,
        }

        response = httpx.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()

        return data.get('links', {}).get('html', {}).get('href', '')

    def download_issues(self) -> list[Any]:
        """Download all issues from the repository.

        Returns:
            A list of issues
        """
        logger.warning('BitbucketIssueHandler.download_issues not implemented')
        return []

    def get_issue_comments(
        self, issue_number: int, comment_id: int | None = None
    ) -> list[str] | None:
        """Get comments for an issue.

        Args:
            issue_number: The issue number
            comment_id: The comment ID (optional)

        Returns:
            A list of comments
        """
        logger.warning('BitbucketIssueHandler.get_issue_comments not implemented')
        return []

    def get_branch_url(self, branch_name: str) -> str:
        """Get the URL for a branch.

        Args:
            branch_name: The branch name

        Returns:
            The URL for the branch
        """
        return (
            f'https://{self.base_domain}/{self.owner}/{self.repo}/branch/{branch_name}'
        )

    def get_compare_url(self, branch_name: str) -> str:
        """Get the URL for comparing branches.

        Args:
            branch_name: The branch name

        Returns:
            The URL for comparing branches
        """
        return f'https://{self.base_domain}/{self.owner}/{self.repo}/compare/master...{branch_name}'

    def get_authorize_url(self) -> str:
        """Get the URL for authorization.

        Returns:
            The URL for authorization
        """
        return f'https://oauth2:{self.token}@{self.base_domain}/'

    def get_pull_url(self, pr_number: int) -> str:
        """Get the URL for a pull request.

        Args:
            pr_number: The pull request number

        Returns:
            The URL for the pull request
        """
        return f'https://{self.base_domain}/{self.owner}/{self.repo}/pull-requests/{pr_number}'

    def get_branch_name(self, base_branch_name: str) -> str:
        """Get a unique branch name.

        Args:
            base_branch_name: The base branch name

        Returns:
            A unique branch name
        """
        return f'{base_branch_name}-{self.owner}'

    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists.

        Args:
            branch_name: The branch name

        Returns:
            True if the branch exists, False otherwise
        """
        logger.warning('BitbucketIssueHandler.branch_exists not implemented')
        return False

    def get_default_branch_name(self) -> str:
        """Get the default branch name.

        Returns:
            The default branch name
        """
        return 'master'

    def create_pull_request(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a pull request.

        Args:
            data: The pull request data

        Returns:
            The created pull request
        """
        if data is None:
            data = {}

        title = data.get('title', '')
        description = data.get('description', '')
        source_branch = data.get('source_branch', '')
        target_branch = data.get('target_branch', '')

        url = f'{self.base_url}/repositories/{self.owner}/{self.repo}/pullrequests'

        payload = {
            'title': title,
            'description': description,
            'source': {'branch': {'name': source_branch}},
            'destination': {'branch': {'name': target_branch}},
            'close_source_branch': False,
        }

        response = httpx.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Ensure data is not None before accessing it
        if data is None:
            data = {}

        return {
            'html_url': data.get('links', {}).get('html', {}).get('href', ''),
            'number': data.get('id', 0),
        }

    def request_reviewers(self, reviewer: str, pr_number: int) -> None:
        """Request reviewers for a pull request.

        Args:
            reviewer: The reviewer
            pr_number: The pull request number
        """
        logger.warning('BitbucketIssueHandler.request_reviewers not implemented')

    def send_comment_msg(self, issue_number: int, msg: str) -> None:
        """Send a comment to an issue.

        Args:
            issue_number: The issue number
            msg: The message
        """
        url = f'{self.base_url}/repositories/{self.owner}/{self.repo}/pullrequests/{issue_number}/comments'

        payload = {'content': {'raw': msg}}

        response = httpx.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

    def get_issue_thread_comments(self, issue_number: int) -> list[str]:
        """Get thread comments for an issue.

        Args:
            issue_number: The issue number

        Returns:
            A list of thread comments
        """
        logger.warning(
            'BitbucketIssueHandler.get_issue_thread_comments not implemented'
        )
        return []

    def get_issue_review_comments(self, issue_number: int) -> list[str]:
        """Get review comments for an issue.

        Args:
            issue_number: The issue number

        Returns:
            A list of review comments
        """
        logger.warning(
            'BitbucketIssueHandler.get_issue_review_comments not implemented'
        )
        return []

    def get_issue_review_threads(self, issue_number: int) -> list[ReviewThread]:
        """Get review threads for an issue.

        Args:
            issue_number: The issue number

        Returns:
            A list of review threads
        """
        logger.warning('BitbucketIssueHandler.get_issue_review_threads not implemented')
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
        """Get context from external issue references.

        Args:
            closing_issues: List of closing issue references
            closing_issue_numbers: List of closing issue numbers
            issue_body: The issue body
            review_comments: List of review comments
            review_threads: List of review threads
            thread_comments: List of thread comments

        Returns:
            Context from external issue references
        """
        new_issue_references = []

        if issue_body:
            new_issue_references.extend(extract_issue_references(issue_body))

        if review_comments:
            for comment in review_comments:
                new_issue_references.extend(extract_issue_references(comment))

        if review_threads:
            for review_thread in review_threads:
                new_issue_references.extend(
                    extract_issue_references(review_thread.comment)
                )

        if thread_comments:
            for thread_comment in thread_comments:
                new_issue_references.extend(extract_issue_references(thread_comment))

        non_duplicate_references = set(new_issue_references)
        unique_issue_references = non_duplicate_references.difference(
            closing_issue_numbers
        )

        for issue_number in unique_issue_references:
            try:
                url = f'{self.base_url}/repositories/{self.owner}/{self.repo}/issues/{issue_number}'
                response = httpx.get(url, headers=self.headers)
                response.raise_for_status()
                issue_data = response.json()
                issue_body = issue_data.get('content', {}).get('raw', '')
                if issue_body:
                    closing_issues.append(issue_body)
            except httpx.HTTPError as e:
                logger.warning(f'Failed to fetch issue {issue_number}: {str(e)}')

        return closing_issues

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        """Get converted issues.

        Args:
            issue_numbers: List of issue numbers
            comment_id: The comment ID

        Returns:
            A list of converted issues
        """
        if not issue_numbers:
            raise ValueError('Unspecified issue numbers')

        all_issues = self.download_issues()
        logger.info(f'Limiting resolving to issues {issue_numbers}.')
        all_issues = [issue for issue in all_issues if issue.get('id') in issue_numbers]

        converted_issues = []
        for issue in all_issues:
            # For PRs, body can be None
            if any([issue.get(key) is None for key in ['id', 'title']]):
                logger.warning(f'Skipping #{issue} as it is missing id or title.')
                continue

            # Handle None body for PRs
            body = (
                issue.get('content', {}).get('raw', '')
                if issue.get('content') is not None
                else ''
            )

            # Placeholder for PR metadata
            closing_issues: list[str] = []
            review_comments: list[str] = []
            review_threads: list[ReviewThread] = []
            thread_ids: list[str] = []
            head_branch = issue.get('source', {}).get('branch', {}).get('name', '')
            thread_comments: list[str] = []

            issue_details = Issue(
                owner=self.owner,
                repo=self.repo,
                number=issue['id'],
                title=issue['title'],
                body=body,
                closing_issues=closing_issues,
                review_comments=review_comments,
                review_threads=review_threads,
                thread_ids=thread_ids,
                head_branch=head_branch,
                thread_comments=thread_comments,
            )

            converted_issues.append(issue_details)

        return converted_issues

    def get_graphql_url(self) -> str:
        """Get the GraphQL URL.

        Returns:
            The GraphQL URL
        """
        return f'https://api.{self.base_domain}/graphql'

    def reply_to_comment(self, pr_number: int, comment_id: str, reply: str) -> None:
        """Reply to a comment.

        Args:
            pr_number: The pull request number
            comment_id: The comment ID
            reply: The reply message
        """
        url = f'{self.base_url}/repositories/{self.owner}/{self.repo}/pullrequests/{pr_number}/comments/{comment_id}'

        payload = {'content': {'raw': reply}}

        response = httpx.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

    def get_issue_references(self, body: str) -> list[int]:
        """Extract issue references from a string.

        Args:
            body: The string to extract issue references from

        Returns:
            A list of issue numbers
        """
        return extract_issue_references(body)


class BitbucketPRHandler(BitbucketIssueHandler):
    """Handler for Bitbucket pull requests, extending the issue handler."""

    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str | None = None,
        base_domain: str = 'bitbucket.org',
    ):
        """Initialize a Bitbucket PR handler.

        Args:
            owner: The workspace of the repository
            repo: The name of the repository
            token: The Bitbucket API token
            username: Optional Bitbucket username
            base_domain: The domain for Bitbucket Server (default: "bitbucket.org")
        """
        super().__init__(owner, repo, token, username, base_domain)
