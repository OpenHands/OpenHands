# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from openhands.core.config import LLMConfig
from openhands.integrations.provider import ProviderType
from openhands.resolver.interfaces.azure_devops import AzureDevOpsIssueHandler
from openhands.resolver.interfaces.bitbucket import (
    BitbucketIssueHandler,
    BitbucketPRHandler,
)
from openhands.resolver.interfaces.forgejo import (
    ForgejoIssueHandler,
    ForgejoPRHandler,
)
from openhands.resolver.interfaces.github import GithubIssueHandler, GithubPRHandler
from openhands.resolver.interfaces.gitlab import GitlabIssueHandler, GitlabPRHandler
from openhands.resolver.interfaces.issue_definitions import (
    ServiceContextIssue,
    ServiceContextPR,
)


class IssueHandlerFactory:
    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str,
        platform: ProviderType,
        base_domain: str,
        issue_type: str,
        llm_config: LLMConfig,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.token = token
        self.username = username
        self.platform = platform
        self.base_domain = base_domain
        self.issue_type = issue_type
        self.llm_config = llm_config

    def create(self) -> ServiceContextIssue | ServiceContextPR:
        if self.issue_type == 'issue':
            if self.platform == ProviderType.GITHUB:
                return ServiceContextIssue(
                    GithubIssueHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm_config,
                )
            elif self.platform == ProviderType.GITLAB:
                return ServiceContextIssue(
                    GitlabIssueHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm_config,
                )
            elif self.platform == ProviderType.BITBUCKET:
                return ServiceContextIssue(
                    BitbucketIssueHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm_config,
                )
            elif self.platform == ProviderType.FORGEJO:
                return ServiceContextIssue(
                    ForgejoIssueHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm_config,
                )
            elif self.platform == ProviderType.AZURE_DEVOPS:
                # Parse owner as organization/project
                parts = self.owner.split('/')
                if len(parts) < 2:
                    raise ValueError(
                        f'Invalid Azure DevOps owner format: {self.owner}. Expected format: organization/project'
                    )

                organization = parts[0]
                project = parts[1]

                return ServiceContextIssue(
                    AzureDevOpsIssueHandler(
                        self.token,
                        organization,
                        project,
                        self.repo,
                    ),
                    self.llm_config,
                )
            else:
                raise ValueError(f'Unsupported platform: {self.platform}')
        elif self.issue_type == 'pr':
            if self.platform == ProviderType.GITHUB:
                return ServiceContextPR(
                    GithubPRHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm_config,
                )
            elif self.platform == ProviderType.GITLAB:
                return ServiceContextPR(
                    GitlabPRHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm_config,
                )
            elif self.platform == ProviderType.BITBUCKET:
                return ServiceContextPR(
                    BitbucketPRHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm_config,
                )
            elif self.platform == ProviderType.FORGEJO:
                return ServiceContextPR(
                    ForgejoPRHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm_config,
                )
            elif self.platform == ProviderType.AZURE_DEVOPS:
                # Parse owner as organization/project
                parts = self.owner.split('/')
                if len(parts) < 2:
                    raise ValueError(
                        f'Invalid Azure DevOps owner format: {self.owner}. Expected format: organization/project'
                    )

                organization = parts[0]
                project = parts[1]

                # For now, use the same handler for both issues and PRs
                return ServiceContextPR(
                    AzureDevOpsIssueHandler(
                        self.token,
                        organization,
                        project,
                        self.repo,
                    ),
                    self.llm_config,
                )
            else:
                raise ValueError(f'Unsupported platform: {self.platform}')
        else:
            raise ValueError(f'Invalid issue type: {self.issue_type}')
