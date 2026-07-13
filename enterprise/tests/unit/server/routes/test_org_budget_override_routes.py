from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from server.routes.orgs import org_router

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
async def test_upsert_org_budget_override_success(mock_app, mock_admin_role):
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    user_row = {
        'user_id': str(user_id),
        'user_email': 'user@example.com',
        'user_name': 'Budget User',
        'current_spend': 12.34,
        'monthly_limit': 50.0,
        'effective_monthly_limit': 50.0,
        'is_disabled': False,
        'is_override': True,
    }

    with (
        patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_admin_role),
        ),
        patch(
            'server.routes.orgs.OrgBudgetService.upsert_user_override',
            AsyncMock(),
        ) as upsert_mock,
        patch(
            'server.routes.orgs.OrgBudgetService.get_user_budget_row',
            AsyncMock(return_value=user_row),
        ) as get_row_mock,
    ):
        client = TestClient(mock_app)
        response = client.put(
            f'/api/organizations/{org_id}/budgets/overrides/{user_id}',
            json={'monthly_limit': 50.0, 'is_disabled': False},
        )

    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    assert response_data['user_id'] == str(user_id)
    assert response_data['monthly_limit'] == 50.0
    upsert_mock.assert_awaited_once_with(
        org_id,
        user_id,
        monthly_limit=50.0,
        is_disabled=False,
    )
    get_row_mock.assert_awaited_once_with(org_id, user_id)


@pytest.mark.asyncio
async def test_delete_org_budget_override_success(mock_app, mock_admin_role):
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()

    with (
        patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_admin_role),
        ),
        patch(
            'server.routes.orgs.OrgBudgetService.delete_user_override',
            AsyncMock(),
        ) as delete_mock,
    ):
        client = TestClient(mock_app)
        response = client.delete(
            f'/api/organizations/{org_id}/budgets/overrides/{user_id}'
        )

    assert response.status_code == status.HTTP_204_NO_CONTENT
    delete_mock.assert_awaited_once_with(org_id, user_id)
