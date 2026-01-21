"""Type definitions for Azure DevOps webhook payloads and API responses."""

from enum import Enum
from typing import TypedDict


class AzureDevOpsEventType(str, Enum):
    """Azure DevOps Service Hook event types."""

    WORKITEM_UPDATED = 'workitem.updated'
    WORKITEM_CREATED = 'workitem.created'
    PR_CREATED = 'git.pullrequest.created'
    PR_UPDATED = 'git.pullrequest.updated'
    PR_COMMENTED = 'ms.vss-code.git-pullrequest-comment-event'


class ResourceContainer(TypedDict):
    """Resource container with ID."""

    id: str


class ResourceContainers(TypedDict):
    """Resource containers in webhook payload."""

    project: ResourceContainer
    account: ResourceContainer
    collection: ResourceContainer


class PullRequestResource(TypedDict, total=False):
    """Pull request resource in webhook payload."""

    repository: dict
    pullRequestId: int
    status: str
    createdBy: dict
    title: str
    description: str
    sourceRefName: str
    targetRefName: str
    url: str


class PullRequestCommentedPayload(TypedDict):
    """Azure DevOps ms.vss-code.git-pullrequest-comment-event webhook payload."""

    id: str
    eventType: str
    publisherId: str
    scope: str
    resource: PullRequestResource
    resourceContainers: ResourceContainers
    createdDate: str


class JsonPatchOperation(TypedDict):
    """JSON Patch operation for work item updates."""

    op: str
    path: str
    value: str


class PRComment(TypedDict):
    """Pull request comment for REST API."""

    parentCommentId: int
    content: str
    commentType: int


class PRCommentThread(TypedDict):
    """Pull request comment thread for REST API."""

    comments: list[PRComment]
    status: int
