import os
from unittest.mock import MagicMock, patch

import pytest
from integrations.jira_dc.jira_dc_service_account import (
    JiraDcServiceAccountConfigError,
    JiraDcServiceAccountResolutionError,
    get_jira_dc_managed_service_account,
    get_jira_dc_service_account_config_error,
    resolve_jira_dc_service_account,
)


def test_resolve_service_account_uses_workspace_credentials(sample_jira_dc_workspace):
    token_manager = MagicMock()
    token_manager.decrypt_text.return_value = 'workspace-pat'

    with patch.dict(
        os.environ,
        {'JIRA_DC_SERVICE_ACCOUNT_EMAIL': '', 'JIRA_DC_SERVICE_ACCOUNT_PAT': ''},
    ):
        service_account = resolve_jira_dc_service_account(
            sample_jira_dc_workspace, token_manager
        )

    assert service_account.email == 'service@company.com'
    assert service_account.api_key == 'workspace-pat'
    assert service_account.managed_by_env is False
    token_manager.decrypt_text.assert_called_once_with('encrypted_api_key')


def test_resolve_service_account_prefers_env_credentials(sample_jira_dc_workspace):
    token_manager = MagicMock()

    with patch.dict(
        os.environ,
        {
            'JIRA_DC_SERVICE_ACCOUNT_EMAIL': 'managed@company.com',
            'JIRA_DC_SERVICE_ACCOUNT_PAT': 'managed-pat',
        },
    ):
        service_account = resolve_jira_dc_service_account(
            sample_jira_dc_workspace, token_manager
        )

    assert service_account.email == 'managed@company.com'
    assert service_account.api_key == 'managed-pat'
    assert service_account.managed_by_env is True
    token_manager.decrypt_text.assert_not_called()


@pytest.mark.parametrize(
    ('email', 'api_key', 'expected_error'),
    [
        ('managed@company.com', '', 'partially configured'),
        ('invalid-email', 'managed-pat', 'valid email'),
        ('managed@company.com', 'managed pat', 'cannot contain whitespace'),
    ],
)
def test_service_account_env_config_errors(email, api_key, expected_error):
    with patch.dict(
        os.environ,
        {
            'JIRA_DC_SERVICE_ACCOUNT_EMAIL': email,
            'JIRA_DC_SERVICE_ACCOUNT_PAT': api_key,
        },
    ):
        assert expected_error in (get_jira_dc_service_account_config_error() or '')


def test_get_managed_service_account_raises_typed_config_error():
    with patch.dict(
        os.environ,
        {
            'JIRA_DC_SERVICE_ACCOUNT_EMAIL': 'managed@company.com',
            'JIRA_DC_SERVICE_ACCOUNT_PAT': 'managed pat',
        },
    ):
        with pytest.raises(JiraDcServiceAccountConfigError):
            get_jira_dc_managed_service_account()


def test_resolve_service_account_raises_typed_resolution_error(
    sample_jira_dc_workspace,
):
    token_manager = MagicMock()
    sample_jira_dc_workspace.svc_acc_api_key = ''

    with patch.dict(
        os.environ,
        {'JIRA_DC_SERVICE_ACCOUNT_EMAIL': '', 'JIRA_DC_SERVICE_ACCOUNT_PAT': ''},
    ):
        with pytest.raises(JiraDcServiceAccountResolutionError):
            resolve_jira_dc_service_account(sample_jira_dc_workspace, token_manager)
