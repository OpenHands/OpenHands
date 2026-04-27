from openhands.integrations.gitlab.service.base import GitLabMixinBase
from openhands.integrations.service_types import (
    ProviderType,
    RequestMethod,
    SuggestedTask,
    TaskType,
)


class GitLabFeaturesMixin(GitLabMixinBase):
    """
    Methods used for custom features in UI driven via GitLab integration
    """

    async def get_suggested_tasks(self) -> list[SuggestedTask]:
        """Get suggested tasks for the authenticated user across all repositories.

        Returns:
        - Merge requests authored by the user.
        - Issues assigned to the user.
        """
        # Get user info to use in queries
        user = await self.get_user()
        username = user.login

        # GraphQL query to get merge requests
        query = """
        query GetUserTasks {
          currentUser {
            authoredMergeRequests(state: opened, sort: UPDATED_DESC, first: 100) {
              nodes {
                id
                iid
                title
                project {
                  fullPath
                }
                conflicts
                mergeStatus
                pipelines(first: 1) {
                  nodes {
                    status
                  }
                }
                discussions(first: 100) {
                  nodes {
                    notes {
                      nodes {
                        resolvable
                        resolved
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """

        try:
            tasks: list[SuggestedTask] = []

            # Get merge requests using GraphQL
            response = await self.execute_graphql_query(query)
            data = response.get('currentUser', {})

            # Process merge requests
            merge_requests = data.get('authoredMergeRequests', {}).get('nodes', [])
            for mr in merge_requests:
                repo_name = mr.get('project', {}).get('fullPath', '')
                mr_number = mr.get('iid')
                title = mr.get('title', '')

                # Start with default task type
                task_type = TaskType.OPEN_PR

                # Check for specific states
                if mr.get('conflicts'):
                    task_type = TaskType.MERGE_CONFLICTS
                elif (
                    mr.get('pipelines', {}).get('nodes', [])
                    and mr.get('pipelines', {}).get('nodes', [])[0].get('status')
                    == 'FAILED'
                ):
                    task_type = TaskType.FAILING_CHECKS
                else:
                    # Check for unresolved comments
                    has_unresolved_comments = False
                    for discussion in mr.get('discussions', {}).get('nodes', []):
                        for note in discussion.get('notes', {}).get('nodes', []):
                            if note.get('resolvable') and not note.get('resolved'):
                                has_unresolved_comments = True
                                break
                        if has_unresolved_comments:
                            break

                    if has_unresolved_comments:
                        task_type = TaskType.UNRESOLVED_COMMENTS

                # Only add the task if it's not OPEN_PR
                if task_type != TaskType.OPEN_PR:
                    tasks.append(
                        SuggestedTask(
                            git_provider=ProviderType.GITLAB,
                            task_type=task_type,
                            repo=repo_name,
                            issue_number=mr_number,
                            title=title,
                        )
                    )

            # Get assigned issues using REST API
            url = f'{self.BASE_URL}/issues'
            params = {
                'assignee_username': username,
                'state': 'opened',
                'scope': 'assigned_to_me',
            }

            issues_response, _ = await self._make_request(
                method=RequestMethod.GET, url=url, params=params
            )

            # Process issues
            for issue in issues_response:
                repo_name = (
                    issue.get('references', {}).get('full', '').split('#')[0].strip()
                )
                issue_number = issue.get('iid')
                title = issue.get('title', '')

                tasks.append(
                    SuggestedTask(
                        git_provider=ProviderType.GITLAB,
                        task_type=TaskType.OPEN_ISSUE,
                        repo=repo_name,
                        issue_number=issue_number,
                        title=title,
                    )
                )

            return tasks
        except Exception:
            return []
