"""REST API query templates and builders for Azure DevOps."""

from urllib.parse import quote

from integrations.azure_devops.azure_devops_types import (
    JsonPatchOperation,
    PRComment,
    PRCommentThread,
)


def build_work_item_comment_patch(comment_text: str) -> list[JsonPatchOperation]:
    """Build JSON-Patch operation to add a comment to a work item.

    Args:
        comment_text: The comment text to add

    Returns:
        List of JSON-Patch operations
    """
    return [{'op': 'add', 'path': '/fields/System.History', 'value': comment_text}]


def build_pr_comment_thread(comment_text: str) -> PRCommentThread:
    """Build PR comment thread payload for REST API.

    Args:
        comment_text: The comment text to add

    Returns:
        PR comment thread payload
    """
    comment: PRComment = {
        'parentCommentId': 0,
        'content': comment_text,
        'commentType': 1,  # 1 = text comment
    }

    return {
        'comments': [comment],
        'status': 1,  # 1 = active
    }


def get_work_item_update_url(
    organization: str, project: str, work_item_id: int, api_version: str = '7.2'
) -> str:
    """Get the URL for updating a work item.

    Args:
        organization: Azure DevOps organization name
        project: Project name
        work_item_id: Work item ID
        api_version: API version (default: 7.2)

    Returns:
        Full URL for PATCH request
    """
    # URL-encode components to handle spaces and special characters
    org_enc = quote(organization, safe='')
    project_enc = quote(project, safe='')
    return f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/wit/workitems/{work_item_id}?api-version={api_version}'


def get_pr_thread_create_url(
    organization: str,
    project: str,
    repository_id: str,
    pull_request_id: int,
    api_version: str = '7.2',
) -> str:
    """Get the URL for creating a PR comment thread.

    Args:
        organization: Azure DevOps organization name
        project: Project name
        repository_id: Repository ID or name
        pull_request_id: Pull request ID
        api_version: API version (default: 7.2)

    Returns:
        Full URL for POST request
    """
    # URL-encode components to handle spaces and special characters
    org_enc = quote(organization, safe='')
    project_enc = quote(project, safe='')
    repo_enc = quote(repository_id, safe='')
    return f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullRequests/{pull_request_id}/threads?api-version={api_version}'


def get_work_item_headers(access_token: str) -> dict[str, str]:
    """Get HTTP headers for work item API requests.

    Args:
        access_token: Azure DevOps access token (PAT or OAuth token)

    Returns:
        Dictionary of HTTP headers
    """
    return {
        'Content-Type': 'application/json-patch+json',
        'Authorization': f'Bearer {access_token}',
    }


def get_pr_headers(access_token: str) -> dict[str, str]:
    """Get HTTP headers for PR API requests.

    Args:
        access_token: Azure DevOps access token (PAT or OAuth token)

    Returns:
        Dictionary of HTTP headers
    """
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}',
    }
