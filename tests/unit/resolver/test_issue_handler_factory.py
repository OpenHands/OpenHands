import pytest
from pydantic import SecretStr

from openhands.core.config import LLMConfig
from openhands.integrations.provider import ProviderType
from openhands.resolver.interfaces.azure_devops import AzureDevOpsIssueHandler
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
from openhands.resolver.issue_handler_factory import IssueHandlerFactory


@pytest.fixture
def llm_config():
    return LLMConfig(
        model='test-model',
        api_key=SecretStr('test-key'),
    )


@pytest.fixture
def factory_params(llm_config):
    return {
        'owner': 'test-owner',
        'repo': 'test-repo',
        'token': 'test-token',
        'username': 'test-user',
        'llm_config': llm_config,
    }


@pytest.fixture
def azure_factory_params(llm_config):
    return {
        'owner': 'test-org/test-project',
        'repo': 'test-repo',
        'token': 'test-token',
        'username': 'test-user',
        'base_domain': 'dev.azure.com',
        'llm_config': llm_config,
    }


test_cases = [
    # platform, issue_type, base_domain, expected_context_type, expected_handler_type, use_azure_params
    (
        ProviderType.GITHUB,
        'issue',
        'github.com',
        ServiceContextIssue,
        GithubIssueHandler,
        False,
    ),
    (
        ProviderType.GITHUB,
        'pr',
        'github.com',
        ServiceContextPR,
        GithubPRHandler,
        False,
    ),
    (
        ProviderType.GITLAB,
        'issue',
        'gitlab.com',
        ServiceContextIssue,
        GitlabIssueHandler,
        False,
    ),
    (
        ProviderType.GITLAB,
        'pr',
        'gitlab.com',
        ServiceContextPR,
        GitlabPRHandler,
        False,
    ),
    (
        ProviderType.FORGEJO,
        'issue',
        'codeberg.org',
        ServiceContextIssue,
        ForgejoIssueHandler,
        False,
    ),
    (
        ProviderType.FORGEJO,
        'pr',
        'codeberg.org',
        ServiceContextPR,
        ForgejoPRHandler,
        False,
    ),
    (
        ProviderType.AZURE_DEVOPS,
        'issue',
        'dev.azure.com',
        ServiceContextIssue,
        AzureDevOpsIssueHandler,
        True,
    ),
    (
        ProviderType.AZURE_DEVOPS,
        'pr',
        'dev.azure.com',
        ServiceContextPR,
        AzureDevOpsIssueHandler,
        True,
    ),
]


@pytest.mark.parametrize(
    'platform,issue_type,base_domain,expected_context_type,expected_handler_type,use_azure_params',
    test_cases,
)
def test_handler_creation(
    factory_params,
    azure_factory_params,
    platform: ProviderType,
    issue_type: str,
    base_domain: str,
    expected_context_type: type,
    expected_handler_type: type,
    use_azure_params: bool,
):
    params = (
        azure_factory_params
        if use_azure_params
        else {**factory_params, 'base_domain': base_domain}
    )
    factory = IssueHandlerFactory(**params, platform=platform, issue_type=issue_type)

    handler = factory.create()

    assert isinstance(handler, expected_context_type)
    assert isinstance(handler._strategy, expected_handler_type)


def test_invalid_issue_type(factory_params):
    factory = IssueHandlerFactory(
        **factory_params,
        platform=ProviderType.GITHUB,
        issue_type='invalid',
        base_domain='github.com',
    )

    with pytest.raises(ValueError, match='Invalid issue type: invalid'):
        factory.create()
