from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from server.routes.orgs import org_router
from server.services.org_budget_service import BudgetCycle

from openhands.app_server.user_auth import get_user_id

TEST_USER_ID = str(uuid.uuid4())


@pytest.fixture
def mock_app():
    app = FastAPI()
    app.include_router(org_router)

    def mock_get_user_id():
        return TEST_USER_ID

    app.dependency_overrides[get_user_id] = mock_get_user_id
    return app


@pytest.fixture
def mock_admin_role():
    mock_role = MagicMock()
    mock_role.name = 'admin'
    return mock_role


@pytest.mark.asyncio
async def test_get_org_budget_settings_success(mock_app, mock_admin_role):
    org_id = uuid.uuid4()
    cycle_start = datetime(2024, 1, 1, tzinfo=UTC)
    cycle_end = datetime(2024, 2, 1, tzinfo=UTC)
    settings = MagicMock(
        enabled=True,
        monthly_limit=200.0,
        litellm_last_sync_at=None,
        litellm_last_sync_status=None,
        litellm_last_sync_error=None,
        reset_day=1,
        slack_channel='#alerts',
        slack_team_id=None,
        default_user_monthly_limit=25.0,
    )
    threshold = MagicMock(id=1, percentage=80, email_enabled=True, slack_enabled=False)
    state = {
        'settings': settings,
        'thresholds': [threshold],
        'cycle': BudgetCycle(start_at=cycle_start, end_at=cycle_end),
        'current_spend': 100.0,
        'users': [
            {
                'user_id': str(uuid.uuid4()),
                'user_email': 'alice@example.com',
                'user_name': 'Alice',
                'current_spend': 10.0,
                'monthly_limit': None,
                'effective_monthly_limit': 25.0,
                'is_disabled': False,
                'is_override': False,
            }
        ],
        'users_total': 1,
        'users_page': 2,
        'users_per_page': 25,
    }

    with (
        patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_admin_role),
        ),
        patch(
            'server.routes.orgs.OrgBudgetService.get_budget_state',
            AsyncMock(return_value=state),
        ) as get_state_mock,
    ):
        client = TestClient(mock_app)
        response = client.get(
            f'/api/organizations/{org_id}/budgets',
            params={
                'users_page': 2,
                'users_per_page': 25,
                'users_search': 'alice',
                'users_status': 'over90',
            },
        )

    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    assert response_data['current_spend_percentage'] == 50.0
    assert response_data['users_page'] == 2
    assert response_data['users_total'] == 1
    assert response_data['thresholds'][0]['percentage'] == 80
    assert response_data['cycle_start_at'].startswith('2024-01-01')
    assert response_data['cycle_end_at'].startswith('2024-02-01')
    get_state_mock.assert_awaited_once_with(
        org_id,
        users_page=2,
        users_per_page=25,
        users_search='alice',
        users_status='over90',
    )


@pytest.mark.asyncio
async def test_update_org_budget_settings_success(mock_app, mock_admin_role):
    org_id = uuid.uuid4()
    cycle_start = datetime(2024, 5, 1, tzinfo=UTC)
    cycle_end = datetime(2024, 6, 1, tzinfo=UTC)
    settings = MagicMock(
        enabled=True,
        monthly_limit=300.0,
        litellm_last_sync_at=None,
        litellm_last_sync_status=None,
        litellm_last_sync_error=None,
        reset_day=15,
        slack_channel='#budget-alerts',
        slack_team_id='T123',
        default_user_monthly_limit=50.0,
    )
    threshold = MagicMock(id=2, percentage=90, email_enabled=True, slack_enabled=True)
    state = {
        'settings': settings,
        'thresholds': [threshold],
        'cycle': BudgetCycle(start_at=cycle_start, end_at=cycle_end),
        'current_spend': 120.0,
        'users': [],
        'users_total': 0,
        'users_page': 1,
        'users_per_page': 25,
    }

    update_payload = {
        'enabled': True,
        'monthly_limit': 300.0,
        'reset_day': 15,
        'default_user_monthly_limit': 50.0,
        'slack_channel': '#budget-alerts',
        'slack_team_id': 'T123',
        'thresholds': [
            {
                'percentage': 90,
                'email_enabled': True,
                'slack_enabled': True,
            }
        ],
    }

    with (
        patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_admin_role),
        ),
        patch(
            'server.routes.orgs.OrgBudgetService.update_budget_settings',
            AsyncMock(return_value=state),
        ) as update_mock,
    ):
        client = TestClient(mock_app)
        response = client.patch(
            f'/api/organizations/{org_id}/budgets',
            params={
                'users_page': 1,
                'users_per_page': 25,
                'users_search': 'budget',
                'users_status': 'over80',
            },
            json=update_payload,
        )

    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    assert response_data['monthly_limit'] == 300.0
    assert response_data['thresholds'][0]['slack_enabled'] is True
    update_args, update_kwargs = update_mock.await_args
    assert update_args[0] == org_id
    update = update_args[1]
    assert update.enabled is True
    assert update.reset_day == 15
    assert update.thresholds[0].percentage == 90
    assert update_kwargs == {
        'users_page': 1,
        'users_per_page': 25,
        'users_search': 'budget',
        'users_status': 'over80',
    }
