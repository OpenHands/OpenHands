from __future__ import annotations

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


class ForgejoIssueHandler(IssueHandlerInterface):
    """Issue handler implementation for Forgejo-based providers (e.g. Codeberg)."""

    API_PREFIX = '/api/v1'

    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str | None = None,
        base_domain: str = 'codeberg.org',
    ):
        self.owner = owner
        self.repo = repo
        self.token = token
        self.username = username
        self.base_domain = base_domain
        self.base_url = self.get_base_url()
        self.download_url = self.get_download_url()
        self.clone_url = self.get_clone_url()
        self.headers = self.get_headers()

    def _api_root(self) -> str:
        return f'https://{self.base_domain}{self.API_PREFIX}'

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def set_owner(self, owner: str) -> None:
        self.owner = owner
        self.base_url = self.get_base_url()
        self.download_url = self.get_download_url()

    def get_headers(self) -> dict[str, str]:
        return {
            'Authorization': f'token {self.token}',
            'Accept': 'application/json',
        }

    def get_base_url(self) -> str:
        return f'{self._api_root()}/repos/{self.owner}/{self.repo}'

    def get_authorize_url(self) -> str:
        credential = (
            f'{self.username}:{self.token}'
            if self.username
            else f'x-auth-token:{self.token}'
        )
        return f'https://{credential}@{self.base_domain}/'

    def get_branch_url(self, branch_name: str) -> str:
        escaped_branch = quote(branch_name, safe='')
        return f'{self.get_base_url()}/branches/{escaped_branch}'

    def get_download_url(self) -> str:
        return f'{self.get_base_url()}/issues'

    def get_clone_url(self) -> str:
        credential = (
            f'{self.username}:{self.token}'
            if self.username
            else f'x-access-token:{self.token}'
        )
        return f'https://{credential}@{self.base_domain}/{self.owner}/{self.repo}.git'

    def get_graphql_url(self) -> str:
        # Forgejo does not expose a GraphQL endpoint.
        return ''

    def get_compare_url(self, branch_name: str) -> str:
        return (
            f'https://{self.base_domain}/{self.owner}/{self.repo}/compare/{branch_name}'
        )

    def download_issues(self) -> list[Any]:
        page = 1
        all_issues: list[Any] = []

        while True:
            params = {'state': 'open', 'limit': '50', 'page': str(page)}
            response = httpx.get(self.download_url, headers=self.headers, params=params)
            response.raise_for_status()
            issues = response.json()

            if not issues:
                break

            if not isinstance(issues, list) or any(
                not isinstance(issue, dict) for issue in issues
            ):
                raise ValueError(
                    'Expected list of dictionaries from Forgejo issues API.'
                )

            all_issues.extend(issues)
            page += 1

        return all_issues

    def get_issue_comments(
        self, issue_number: int, comment_id: int | None = None
    ) -> list[str] | None:
        url = f'{self.get_download_url()}/{issue_number}/comments'
        page = 1
        params = {'limit': '50', 'page': str(page)}
        all_comments: list[str] = []

        while True:
            response = httpx.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            comments = response.json()

            if not comments:
                break

            if comment_id is not None:
                matching_comment = next(
                    (
                        comment['body']
                        for comment in comments
                        if self._to_int(comment.get('id')) == comment_id
                    ),
                    None,
                )
                if matching_comment:
                    return [matching_comment]
            else:
                all_comments.extend(
                    comment['body'] for comment in comments if comment.get('body')
                )

            page += 1
            params = {'limit': '50', 'page': str(page)}

        return all_comments if all_comments else None

    def get_pull_url(self, pr_number: int) -> str:
        return f'https://{self.base_domain}/{self.owner}/{self.repo}/pulls/{pr_number}'

    def get_branch_name(self, base_branch_name: str) -> str:
        branch_name = base_branch_name
        attempt = 1
        while self.branch_exists(branch_name):
            attempt += 1
            branch_name = f'{base_branch_name}-try{attempt}'
        return branch_name

    def get_default_branch_name(self) -> str:
        response = httpx.get(self.get_base_url(), headers=self.headers)
        response.raise_for_status()
        data = response.json()
        return str(data.get('default_branch'))

    def branch_exists(self, branch_name: str) -> bool:
        response = httpx.get(self.get_branch_url(branch_name), headers=self.headers)
        exists = response.status_code == 200
        logger.info(f'Branch {branch_name} exists: {exists}')
        return exists

    def reply_to_comment(self, pr_number: int, comment_id: str, reply: str) -> None:
        # Forgejo does not support threaded replies via API; add a regular comment referencing the original ID.
        message = f'OpenHands reply to comment {comment_id}\n\n{reply}'
        self.send_comment_msg(pr_number, message)

    def create_pull_request(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = data or {}
        response = httpx.post(
            f'{self.get_base_url()}/pulls', headers=self.headers, json=payload
        )
        if response.status_code == 403:
            raise RuntimeError(
                'Failed to create pull request due to missing permissions. '
                'Ensure the token has write access to the repository.'
            )
        response.raise_for_status()
        pr_data = response.json()
        pr_data.setdefault('number', pr_data.get('index'))
        if 'html_url' not in pr_data and 'url' in pr_data:
            pr_data['html_url'] = pr_data['url']
        return dict(pr_data)

    def request_reviewers(self, reviewer: str, pr_number: int) -> None:
        url = f'{self.get_base_url()}/pulls/{pr_number}/requested_reviewers'
        response = httpx.post(
            url,
            headers=self.headers,
            json={'reviewers': [reviewer]},
        )
        if response.status_code not in (200, 201, 204):
            logger.warning(
                f'Failed to request review from {reviewer}: {response.status_code} {response.text}'
            )

    def send_comment_msg(self, issue_number: int, msg: str) -> None:
        comment_url = f'{self.get_download_url()}/{issue_number}/comments'
        response = httpx.post(
            comment_url,
            headers=self.headers,
            json={'body': msg},
        )
        if response.status_code not in (200, 201):
            logger.error(
                f'Failed to post comment: {response.status_code} {response.text}'
            )

    def get_context_from_external_issues_references(
        self,
        closing_issues: list[str],
        closing_issue_numbers: list[int],
        issue_body: str,
        review_comments: list[str] | None,
        review_threads: list[ReviewThread],
        thread_comments: list[str] | None,
    ) -> list[str]:
        new_references: list[int] = []

        if issue_body:
            new_references.extend(extract_issue_references(issue_body))

        if review_comments:
            for comment in review_comments:
                new_references.extend(extract_issue_references(comment))

        if review_threads:
            for thread in review_threads:
                new_references.extend(extract_issue_references(thread.comment))

        if thread_comments:
            for thread_comment in thread_comments:
                new_references.extend(extract_issue_references(thread_comment))

        unique_ids = set(new_references).difference(closing_issue_numbers)

        for issue_number in unique_ids:
            try:
                response = httpx.get(
                    f'{self.get_download_url()}/{issue_number}',
                    headers=self.headers,
                )
                response.raise_for_status()
                issue_data = response.json()
                body = issue_data.get('body', '')
                if body:
                    closing_issues.append(body)
            except httpx.HTTPError as exc:
                logger.warning(f'Failed to fetch issue {issue_number}: {exc}')

        return closing_issues

    def get_pull_url_for_issue(self, issue_number: int) -> str:
        return (
            f'https://{self.base_domain}/{self.owner}/{self.repo}/issues/{issue_number}'
        )

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        if not issue_numbers:
            raise ValueError('Unspecified issue numbers')

        all_issues = self.download_issues()
        logger.info(f'Limiting resolving to issues {issue_numbers}.')
        filtered = [
            issue
            for issue in all_issues
            if self._to_int(issue.get('number') or issue.get('index')) in issue_numbers
        ]

        converted: list[Issue] = []
        for issue in filtered:
            if any(issue.get(key) is None for key in ['number', 'title']):
                logger.warning(
                    f'Skipping issue {issue} as it is missing number or title.'
                )
                continue

            issue_number = self._to_int(issue.get('number') or issue.get('index'))
            body = issue.get('body') or ''
            thread_comments = self.get_issue_comments(issue_number, comment_id)

            issue_details = Issue(
                owner=self.owner,
                repo=self.repo,
                number=issue_number,
                title=issue['title'],
                body=body,
                thread_comments=thread_comments,
                review_comments=None,
                review_threads=None,
            )
            converted.append(issue_details)

        return converted


class ForgejoPRHandler(ForgejoIssueHandler):
    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str | None = None,
        base_domain: str = 'codeberg.org',
    ):
        super().__init__(owner, repo, token, username, base_domain)
        self.download_url = f'{self.get_base_url()}/pulls'

    def download_pr_metadata(
        self, pull_number: int, comment_id: int | None = None
    ) -> tuple[list[str], list[int], list[str] | None, list[ReviewThread], list[str]]:
        closing_issues: list[str] = []
        closing_issue_numbers: list[int] = []

        try:
            response = httpx.get(
                f'{self.get_base_url()}/pulls/{pull_number}', headers=self.headers
            )
            response.raise_for_status()
            pr_data = response.json()
            body = pr_data.get('body') or ''
            closing_refs = extract_issue_references(body)
            closing_issue_numbers.extend(closing_refs)
            if body:
                closing_issues.append(body)
        except httpx.HTTPError as exc:
            logger.warning(f'Failed to fetch PR metadata for {pull_number}: {exc}')

        review_comments = self.get_pr_comments(pull_number, comment_id)
        review_threads: list[ReviewThread] = []
        thread_ids: list[str] = []

        return (
            closing_issues,
            closing_issue_numbers,
            review_comments,
            review_threads,
            thread_ids,
        )

    def get_pr_comments(
        self, pr_number: int, comment_id: int | None = None
    ) -> list[str] | None:
        url = f'{self.get_base_url()}/pulls/{pr_number}/comments'
        page = 1
        params = {'limit': '50', 'page': str(page)}
        collected: list[str] = []

        while True:
            response = httpx.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            comments = response.json()

            if not comments:
                break

            filtered = [
                comment for comment in comments if not comment.get('is_system', False)
            ]

            if comment_id is not None:
                matching = next(
                    (
                        comment['body']
                        for comment in filtered
                        if self._to_int(comment.get('id')) == comment_id
                    ),
                    None,
                )
                if matching:
                    return [matching]
            else:
                collected.extend(
                    comment['body'] for comment in filtered if comment.get('body')
                )

            page += 1
            params = {'limit': '50', 'page': str(page)}

        return collected if collected else None

    def get_context_from_external_issues_references(
        self,
        closing_issues: list[str],
        closing_issue_numbers: list[int],
        issue_body: str,
        review_comments: list[str] | None,
        review_threads: list[ReviewThread],
        thread_comments: list[str] | None,
    ) -> list[str]:
        return super().get_context_from_external_issues_references(
            closing_issues,
            closing_issue_numbers,
            issue_body,
            review_comments,
            review_threads,
            thread_comments,
        )

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        if not issue_numbers:
            raise ValueError('Unspecified issue numbers')

        response = httpx.get(self.download_url, headers=self.headers)
        response.raise_for_status()
        all_prs = response.json()

        logger.info(f'Limiting resolving to PRs {issue_numbers}.')
        filtered = [
            pr
            for pr in all_prs
            if self._to_int(pr.get('number') or pr.get('index')) in issue_numbers
        ]

        converted: list[Issue] = []
        for pr in filtered:
            if any(pr.get(key) is None for key in ['number', 'title']):
                logger.warning(f'Skipping PR {pr} as it is missing number or title.')
                continue

            body = pr.get('body') or ''
            pr_number = self._to_int(pr.get('number') or pr.get('index', 0))
            (
                closing_issues,
                closing_issue_numbers,
                review_comments,
                review_threads,
                thread_ids,
            ) = self.download_pr_metadata(pr_number, comment_id)
            head_branch = (pr.get('head') or {}).get('ref')
            thread_comments = self.get_pr_comments(pr_number, comment_id)

            closing_issues = self.get_context_from_external_issues_references(
                closing_issues,
                closing_issue_numbers,
                body,
                review_comments,
                review_threads,
                thread_comments,
            )

            issue_details = Issue(
                owner=self.owner,
                repo=self.repo,
                number=pr_number,
                title=pr['title'],
                body=body,
                closing_issues=closing_issues,
                review_comments=review_comments,
                review_threads=review_threads,
                thread_ids=thread_ids,
                head_branch=head_branch,
                thread_comments=thread_comments,
            )

            converted.append(issue_details)

        return converted
