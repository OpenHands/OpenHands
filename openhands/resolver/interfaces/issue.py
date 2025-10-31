from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ReviewThread(BaseModel):
    comment: str
    files: list[str]


class Issue(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    body: str
    thread_comments: list[str] | None = None  # Added field for issue thread comments
    closing_issues: list[str] | None = None
    review_comments: list[str] | None = None
    review_threads: list[ReviewThread] | None = None
    thread_ids: list[str] | None = None
    head_branch: str | None = None
    base_branch: str | None = None


class IssueHandlerInterface(ABC):
    @abstractmethod
    def set_owner(self, owner: str) -> None:
        pass

    @abstractmethod
    def download_issues(self) -> list[Any]:
        pass

    @abstractmethod
    def get_issue_comments(
        self, issue_number: int, comment_id: int | None = None
    ) -> list[str] | None:
        pass

    @abstractmethod
    def get_base_url(self) -> str:
        pass

    @abstractmethod
    def get_branch_url(self, branch_name: str) -> str:
        pass

    @abstractmethod
    def get_download_url(self) -> str:
        pass

    @abstractmethod
    def get_clone_url(self) -> str:
        pass

    @abstractmethod
    def get_pull_url(self, pr_number: int) -> str:
        pass

    @abstractmethod
    def get_graphql_url(self) -> str:
        pass

    @abstractmethod
    def get_headers(self) -> dict[str, str]:
        pass

    @abstractmethod
    def get_compare_url(self, branch_name: str) -> str:
        pass

    @abstractmethod
    def get_branch_name(self, base_branch_name: str) -> str:
        pass

    @abstractmethod
    def get_default_branch_name(self) -> str:
        pass

    @abstractmethod
    def branch_exists(self, branch_name: str) -> bool:
        pass

    @abstractmethod
    def reply_to_comment(self, pr_number: int, comment_id: str, reply: str) -> None:
        pass

    @abstractmethod
    def send_comment_msg(self, issue_number: int, msg: str) -> None:
        pass

    @abstractmethod
    def get_authorize_url(self) -> str:
        pass

    @abstractmethod
    def create_pull_request(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if data is None:
            data = {}
        raise NotImplementedError

    @abstractmethod
    def request_reviewers(self, reviewer: str, pr_number: int) -> None:
        pass

    @abstractmethod
    def get_context_from_external_issues_references(
        self,
        closing_issues: list[str],
        closing_issue_numbers: list[int],
        issue_body: str,
        review_comments: list[str] | None,
        review_threads: list[ReviewThread],
        thread_comments: list[str] | None,
    ) -> list[str]:
        pass

    @abstractmethod
    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        """Download issues from the git provider (GitHub, GitLab, Bitbucket, or Azure DevOps)."""
        pass
