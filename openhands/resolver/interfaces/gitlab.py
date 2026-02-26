# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
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


class GitlabIssueHandler(IssueHandlerInterface):
    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str | None = None,
        base_domain: str = 'gitlab.com',
    ):
        """Initialize a GitLab issue handler.

        Args:
            owner: The owner of the repository
            repo: The name of the repository
            token: The GitLab personal access token
            username: Optional GitLab username
            base_domain: The domain for GitLab Enterprise (default: "gitlab.com")
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
        return {
            'Authorization': f'Bearer {self.token}',
            'Accept': 'application/json',
        }

    def get_base_url(self) -> str:
        project_path = quote(f'{self.owner}/{self.repo}', safe='')
        return f'https://{self.base_domain}/api/v4/projects/{project_path}'

    def get_authorize_url(self) -> str:
        return f'https://{self.username}:{self.token}@{self.base_domain}/'

    def get_branch_url(self, branch_name: str) -> str:
        return self.get_base_url() + f'/repository/branches/{branch_name}'

    def get_download_url(self) -> str:
        return f'{self.base_url}/issues'

    def get_clone_url(self) -> str:
        username_and_token = self.token
        if self.username:
            username_and_token = f'{self.username}:{self.token}'
        return f'https://{username_and_token}@{self.base_domain}/{self.owner}/{self.repo}.git'

    def get_graphql_url(self) -> str:
        return f'https://{self.base_domain}/api/graphql'

    def get_compare_url(self, branch_name: str) -> str:
        return f'https://{self.base_domain}/{self.owner}/{self.repo}/-/compare/{self.get_default_branch_name()}...{branch_name}'

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        """Download issues from Gitlab.

        Args:
            issue_numbers: The numbers of the issues to download
            comment_id: The ID of a single comment, if provided, otherwise all comments

        Returns:
            List of Gitlab issues.
        """
        if not issue_numbers:
            raise ValueError('Unspecified issue number')

        all_issues = self.download_issues()
        logger.info(f'Limiting resolving to issues {issue_numbers}.')
        all_issues = [
            issue
            for issue in all_issues
            # if issue['iid'] in issue_numbers and issue['merge_requests_count'] == 0
            if issue['iid'] in issue_numbers  # TODO for testing
        ]

        if len(issue_numbers) == 1 and not all_issues:
            raise ValueError(f'Issue {issue_numbers[0]} not found')

        converted_issues = []
        for issue in all_issues:
            if any([issue.get(key) is None for key in ['iid', 'title']]):
                logger.warning(f'Skipping issue {issue} as it is missing iid or title.')
                continue

            # Handle empty body by using empty string
            if issue.get('description') is None:
                issue['description'] = ''

            # Get issue thread comments
            thread_comments = self.get_issue_comments(
                issue['iid'], comment_id=comment_id
            )
            # Convert empty lists to None for optional fields
            issue_details = Issue(
                owner=self.owner,
                repo=self.repo,
                number=issue['iid'],
                title=issue['title'],
                body=issue['description'],
                thread_comments=thread_comments,
                review_comments=None,  # Initialize review comments as None for regular issues
            )

            converted_issues.append(issue_details)

        return converted_issues

    def download_issues(self) -> list[Any]:
        params: dict[str, int | str] = {
            'state': 'opened',
            'scope': 'all',
            'per_page': 100,
            'page': 1,
        }
        all_issues = []

        while True:
            response = httpx.get(self.download_url, headers=self.headers, params=params)
            response.raise_for_status()
            issues = response.json()

            if not issues:
                break

            if not isinstance(issues, list) or any(
                [not isinstance(issue, dict) for issue in issues]
            ):
                raise ValueError(
                    'Expected list of dictionaries from Service Gitlab API.'
                )

            all_issues.extend(issues)
            assert isinstance(params['page'], int)
            params['page'] += 1

        return all_issues

    def get_issue_comments(
        self, issue_number: int, comment_id: int | None = None
    ) -> list[str] | None:
        """Download comments for a specific issue from Gitlab."""
        url = f'{self.download_url}/{issue_number}/notes'
        params = {'per_page': 100, 'page': 1}
        all_comments = []

        while True:
            response = httpx.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            comments = response.json()

            if not comments:
                break

            if comment_id:
                matching_comment = next(
                    (
                        comment['body']
                        for comment in comments
                        if comment['id'] == comment_id
                    ),
                    None,
                )
                if matching_comment:
                    return [matching_comment]
            else:
                all_comments.extend([comment['body'] for comment in comments])

            params['page'] += 1

        return all_comments if all_comments else None

    def branch_exists(self, branch_name: str) -> bool:
        logger.info(f'Checking if branch {branch_name} exists...')
        response = httpx.get(
            f'{self.base_url}/repository/branches/{branch_name}', headers=self.headers
        )
        exists = response.status_code == 200
        logger.info(f'Branch {branch_name} exists: {exists}')
        return exists

    def get_branch_name(self, base_branch_name: str) -> str:
        branch_name = base_branch_name
        attempt = 1
        while self.branch_exists(branch_name):
            attempt += 1
            branch_name = f'{base_branch_name}-try{attempt}'
        return branch_name

    def reply_to_comment(self, pr_number: int, comment_id: str, reply: str) -> None:
        response = httpx.get(
            f'{self.base_url}/merge_requests/{pr_number}/discussions/{comment_id.split("/")[-1]}',
            headers=self.headers,
        )
        response.raise_for_status()
        discussions = response.json()
        if len(discussions.get('notes', [])) > 0:
            data = {
                'body': f'Openhands fix success summary\n\n\n{reply}',
                'note_id': discussions.get('notes', [])[-1]['id'],
            }
            response = httpx.post(
                f'{self.base_url}/merge_requests/{pr_number}/discussions/{comment_id.split("/")[-1]}/notes',
                headers=self.headers,
                json=data,
            )
            response.raise_for_status()

    def get_pull_url(self, pr_number: int) -> str:
        return f'https://{self.base_domain}/{self.owner}/{self.repo}/-/merge_requests/{pr_number}'

    def get_default_branch_name(self) -> str:
        response = httpx.get(f'{self.base_url}', headers=self.headers)
        response.raise_for_status()
        data = response.json()
        return str(data['default_branch'])

    def create_pull_request(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if data is None:
            data = {}
        response = httpx.post(
            f'{self.base_url}/merge_requests', headers=self.headers, json=data
        )
        if response.status_code == 403:
            raise RuntimeError(
                'Failed to create pull request due to missing permissions. '
                'Make sure that the provided token has push permissions for the repository.'
            )
        response.raise_for_status()
        pr_data = response.json()
        if 'web_url' in pr_data:
            pr_data['html_url'] = pr_data['web_url']

        if 'iid' in pr_data:
            pr_data['number'] = pr_data['iid']

        return dict(pr_data)

    def request_reviewers(self, reviewer: str, pr_number: int) -> None:
        response = httpx.get(
            f'https://{self.base_domain}/api/v4/users?username={reviewer}',
            headers=self.headers,
        )
        response.raise_for_status()
        user_data = response.json()
        if len(user_data) > 0:
            review_data = {'reviewer_ids': [user_data[0]['id']]}
            review_response = httpx.put(
                f'{self.base_url}/merge_requests/{pr_number}',
                headers=self.headers,
                json=review_data,
            )
            if review_response.status_code != 200:
                logger.warning(
                    f'Failed to request review from {reviewer}: {review_response.text}'
                )

    def send_comment_msg(self, issue_number: int, msg: str) -> None:
        """Send a comment message to a GitHub issue or pull request.

        Args:
            issue_number: The issue or pull request number
            msg: The message content to post as a comment
        """
        # Post a comment on the PR
        comment_url = f'{self.base_url}/issues/{issue_number}/notes'
        comment_data = {'body': msg}
        comment_response = httpx.post(
            comment_url, headers=self.headers, json=comment_data
        )
        if comment_response.status_code != 201:
            logger.error(
                f'Failed to post comment: {comment_response.status_code} {comment_response.text}'
            )
        else:
            logger.info(f'Comment added to the PR: {msg}')

    def get_context_from_external_issues_references(
        self,
        closing_issues: list[str],
        closing_issue_numbers: list[int],
        issue_body: str,
        review_comments: list[str] | None,
        review_threads: list[ReviewThread],
        thread_comments: list[str] | None,
    ) -> list[str]:
        return []


class GitlabPRHandler(GitlabIssueHandler):
    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str | None = None,
        base_domain: str = 'gitlab.com',
    ):
        """Initialize a GitLab PR handler.

        Args:
            owner: The owner of the repository
            repo: The name of the repository
            token: The GitLab personal access token
            username: Optional GitLab username
            base_domain: The domain for GitLab Enterprise (default: "gitlab.com")
        """
        super().__init__(owner, repo, token, username, base_domain)
        self.download_url = f'{self.base_url}/merge_requests'

    def download_pr_metadata(
        self, pull_number: int, comment_id: int | None = None
    ) -> tuple[list[str], list[int], list[str] | None, list[ReviewThread], list[str]]:
        """Run a GraphQL query against the Gitlab API for information.

        Retrieves information about:
            1. unresolved review comments
            2. referenced issues the pull request would close

        Args:
            pull_number: The number of the pull request to query.
            comment_id: Optional ID of a specific comment to focus on.
            query: The GraphQL query as a string.
            variables: A dictionary of variables for the query.
            token: Your Gitlab personal access token.

        Returns:
            The JSON response from the Gitlab API.
        """
        # Using graphql as REST API doesn't indicate resolved status for review comments
        # TODO: grabbing the first 10 issues, 100 review threads, and 100 coments; add pagination to retrieve all
        response = httpx.get(
            f'{self.base_url}/merge_requests/{pull_number}/related_issues',
            headers=self.headers,
        )
        response.raise_for_status()
        closing_issues = response.json()
        closing_issues_bodies = [issue['description'] for issue in closing_issues]
        closing_issue_numbers = [
            issue['iid'] for issue in closing_issues
        ]  # Extract issue numbers

        query = """
                query($projectPath: ID!, $pr: String!) {
                    project(fullPath: $projectPath) {
                        mergeRequest(iid: $pr) {
                            webUrl
                            discussions(first: 100) {
                                edges {
                                    node {
                                        id
                                        resolved
                                        resolvable
                                        notes(first: 100) {
                                            nodes {
                                                body
                                                id
                                                position {
                                                    filePath
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            """

        project_path = f'{self.owner}/{self.repo}'
        variables = {'projectPath': project_path, 'pr': str(pull_number)}

        response = httpx.post(
            self.get_graphql_url(),
            json={'query': query, 'variables': variables},
            headers=self.headers,
        )
        response.raise_for_status()
        response_json = response.json()

        # Parse the response to get closing issue references and unresolved review comments
        pr_data = (
            response_json.get('data', {}).get('project', {}).get('mergeRequest', {})
        )

        # Get review comments
        review_bodies = None

        # Get unresolved review threads
        review_threads = []
        thread_ids = []  # Store thread IDs; agent replies to the thread
        raw_review_threads = pr_data.get('discussions', {}).get('edges', [])

        for thread in raw_review_threads:
            node = thread.get('node', {})
            if not node.get('resolved', True) and node.get(
                'resolvable', True
            ):  # Check if the review thread is unresolved
                id = node.get('id')
                thread_contains_comment_id = False
                my_review_threads = node.get('notes', {}).get('nodes', [])
                message = ''
                files = []
                for i, review_thread in enumerate(my_review_threads):
                    if (
                        comment_id is not None
                        and int(review_thread['id'].split('/')[-1]) == comment_id
                    ):
                        thread_contains_comment_id = True

                    if (
                        i == len(my_review_threads) - 1
                    ):  # Check if it's the last thread in the thread
                        if len(my_review_threads) > 1:
                            message += '---\n'  # Add "---" before the last message if there's more than one thread
                        message += 'latest feedback:\n' + review_thread['body'] + '\n'
                    else:
                        message += (
                            review_thread['body'] + '\n'
                        )  # Add each thread in a new line

                    file = review_thread.get('position', {})
                    file = file.get('filePath') if file is not None else None
                    if file and file not in files:
                        files.append(file)

                if comment_id is None or thread_contains_comment_id:
                    unresolved_thread = ReviewThread(comment=message, files=files)
                    review_threads.append(unresolved_thread)
                    thread_ids.append(id)

        return (
            closing_issues_bodies,
            closing_issue_numbers,
            review_bodies,
            review_threads,
            thread_ids,
        )

    # Override processing of downloaded issues
    def get_pr_comments(
        self, pr_number: int, comment_id: int | None = None
    ) -> list[str] | None:
        """Download comments for a specific pull request from Gitlab."""
        url = f'{self.base_url}/merge_requests/{pr_number}/notes'
        params = {'per_page': 100, 'page': 1}
        all_comments = []

        while True:
            response = httpx.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            comments = response.json()
            comments = [
                comment
                for comment in comments
                if comment.get('resolvable', True) and not comment.get('system', True)
            ]

            if not comments:
                break

            if comment_id is not None:
                matching_comment = next(
                    (
                        comment['body']
                        for comment in comments
                        if comment['id'] == comment_id
                    ),
                    None,
                )
                if matching_comment:
                    return [matching_comment]
            else:
                all_comments.extend([comment['body'] for comment in comments])

            params['page'] += 1

        return all_comments if all_comments else None

    def get_context_from_external_issues_references(
        self,
        closing_issues: list[str],
        closing_issue_numbers: list[int],
        issue_body: str,
        review_comments: list[str] | None,
        review_threads: list[ReviewThread],
        thread_comments: list[str] | None,
    ) -> list[str]:
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
                url = f'{self.base_url}/issues/{issue_number}'
                response = httpx.get(url, headers=self.headers)
                response.raise_for_status()
                issue_data = response.json()
                issue_body = issue_data.get('description', '')
                if issue_body:
                    closing_issues.append(issue_body)
            except httpx.HTTPError as e:
                logger.warning(f'Failed to fetch issue {issue_number}: {str(e)}')

        return closing_issues

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        if not issue_numbers:
            raise ValueError('Unspecified issue numbers')

        all_issues = self.download_issues()
        logger.info(f'Limiting resolving to issues {issue_numbers}.')
        all_issues = [issue for issue in all_issues if issue['iid'] in issue_numbers]

        converted_issues = []
        for issue in all_issues:
            # For PRs, body can be None
            if any([issue.get(key) is None for key in ['iid', 'title']]):
                logger.warning(f'Skipping #{issue} as it is missing iid or title.')
                continue

            # Handle None body for PRs
            body = (
                issue.get('description') if issue.get('description') is not None else ''
            )
            (
                closing_issues,
                closing_issues_numbers,
                review_comments,
                review_threads,
                thread_ids,
            ) = self.download_pr_metadata(issue['iid'], comment_id=comment_id)
            head_branch = issue['source_branch']

            # Get PR thread comments
            thread_comments = self.get_pr_comments(issue['iid'], comment_id=comment_id)

            closing_issues = self.get_context_from_external_issues_references(
                closing_issues,
                closing_issues_numbers,
                body,
                review_comments,
                review_threads,
                thread_comments,
            )

            issue_details = Issue(
                owner=self.owner,
                repo=self.repo,
                number=issue['iid'],
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
