"""Branch operations for Azure DevOps integration."""

from openhands.integrations.azure_devops.service.base import AzureDevOpsMixinBase
from openhands.integrations.service_types import Branch, PaginatedBranchesResponse


class AzureDevOpsBranchesMixin(AzureDevOpsMixinBase):
    """Mixin for Azure DevOps branch operations."""

    async def get_branches(self, repository: str) -> list[Branch]:
        """Get branches for a repository."""
        # Parse repository string: organization/project/repo
        parts = repository.split('/')
        if len(parts) < 3:
            raise ValueError(
                f'Invalid repository format: {repository}. Expected format: organization/project/repo'
            )

        org = parts[0]
        project = parts[1]
        repo_name = parts[2]

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo_name)

        url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/refs?api-version=7.1&filter=heads/'

        # Set maximum branches to fetch
        MAX_BRANCHES = 1000

        response, _ = await self._make_request(url)
        branches_data = response.get('value', [])

        all_branches = []

        for branch_data in branches_data:
            # Extract branch name from the ref (e.g., "refs/heads/main" -> "main")
            name = branch_data.get('name', '').replace('refs/heads/', '')

            # Get the commit details for this branch
            object_id = branch_data.get('objectId', '')
            commit_url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/commits/{object_id}?api-version=7.1'
            commit_data, _ = await self._make_request(commit_url)

            # Check if the branch is protected
            name_enc = self._encode_url_component(name)
            policy_url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/policy/configurations?api-version=7.1&repositoryId={repo_enc}&refName=refs/heads/{name_enc}'
            policy_data, _ = await self._make_request(policy_url)
            is_protected = len(policy_data.get('value', [])) > 0

            branch = Branch(
                name=name,
                commit_sha=object_id,
                protected=is_protected,
                last_push_date=commit_data.get('committer', {}).get('date'),
            )
            all_branches.append(branch)

            if len(all_branches) >= MAX_BRANCHES:
                break

        return all_branches

    async def get_paginated_branches(
        self, repository: str, page: int = 1, per_page: int = 30
    ) -> PaginatedBranchesResponse:
        """Get branches for a repository with pagination."""
        # Parse repository string: organization/project/repo
        parts = repository.split('/')
        if len(parts) < 3:
            raise ValueError(
                f'Invalid repository format: {repository}. Expected format: organization/project/repo'
            )

        org = parts[0]
        project = parts[1]
        repo_name = parts[2]

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo_name)

        # First, get the repository to get its ID
        repo_url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}?api-version=7.1'
        repo_data, _ = await self._make_request(repo_url)
        repo_id = repo_data.get(
            'id', repo_name
        )  # Fall back to repo_name if ID not found

        url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/refs?api-version=7.1&filter=heads/'

        response, _ = await self._make_request(url)
        branches_data = response.get('value', [])

        # Calculate pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_data = branches_data[start_idx:end_idx]

        branches: list[Branch] = []
        for branch_data in paginated_data:
            # Extract branch name from the ref (e.g., "refs/heads/main" -> "main")
            name = branch_data.get('name', '').replace('refs/heads/', '')

            # Get the commit details for this branch
            object_id = branch_data.get('objectId', '')
            commit_url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/commits/{object_id}?api-version=7.1'
            commit_data, _ = await self._make_request(commit_url)

            # Check if the branch is protected using repository ID
            name_enc = self._encode_url_component(name)
            policy_url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/policy/configurations?api-version=7.1&repositoryId={repo_id}&refName=refs/heads/{name_enc}'
            policy_data, _ = await self._make_request(policy_url)
            is_protected = len(policy_data.get('value', [])) > 0

            branch = Branch(
                name=name,
                commit_sha=object_id,
                protected=is_protected,
                last_push_date=commit_data.get('committer', {}).get('date'),
            )
            branches.append(branch)

        # Determine if there's a next page
        has_next_page = end_idx < len(branches_data)

        return PaginatedBranchesResponse(
            branches=branches,
            has_next_page=has_next_page,
            current_page=page,
            per_page=per_page,
        )

    async def search_branches(
        self, repository: str, query: str, per_page: int = 30
    ) -> list[Branch]:
        """Search for branches within a repository."""
        # Parse repository string: organization/project/repo
        parts = repository.split('/')
        if len(parts) < 3:
            raise ValueError(
                f'Invalid repository format: {repository}. Expected format: organization/project/repo'
            )

        org = parts[0]
        project = parts[1]
        repo_name = parts[2]

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo_name)

        url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/refs?api-version=7.1&filter=heads/'

        try:
            response, _ = await self._make_request(url)
            branches_data = response.get('value', [])

            # Filter branches by query
            filtered_branches = []
            for branch_data in branches_data:
                # Extract branch name from the ref (e.g., "refs/heads/main" -> "main")
                name = branch_data.get('name', '').replace('refs/heads/', '')

                # Check if query matches branch name
                if query.lower() in name.lower():
                    object_id = branch_data.get('objectId', '')

                    # Get commit details for this branch
                    commit_url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/commits/{object_id}?api-version=7.1'
                    try:
                        commit_data, _ = await self._make_request(commit_url)
                        last_push_date = commit_data.get('committer', {}).get('date')
                    except Exception:
                        last_push_date = None

                    branch = Branch(
                        name=name,
                        commit_sha=object_id,
                        protected=False,  # Skip protected check for search to improve performance
                        last_push_date=last_push_date,
                    )
                    filtered_branches.append(branch)

                    if len(filtered_branches) >= per_page:
                        break

            return filtered_branches
        except Exception:
            # Return empty list on error instead of None
            return []
