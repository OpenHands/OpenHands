from __future__ import annotations

import importlib.util
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from storage.jira_dc_integration_store import JiraDcIntegrationStore
from storage.jira_dc_user import JiraDcUser


def _load_migration_module():
    migration_path = (
        Path(__file__).parents[3]
        / 'migrations'
        / 'versions'
        / '115_enforce_single_active_jira_dc_user_link.py'
    )
    spec = importlib.util.spec_from_file_location(
        'migration_115_enforce_single_active_jira_dc_user_link', migration_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_active_link_migration_deduplicates_and_enforces_one_active_link_per_user():
    engine = create_engine('sqlite:///:memory:')
    metadata = sa.MetaData()
    jira_dc_users = sa.Table(
        'jira_dc_users',
        metadata,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('keycloak_user_id', sa.String, nullable=False),
        sa.Column('jira_dc_user_id', sa.String, nullable=False),
        sa.Column('jira_dc_workspace_id', sa.Integer, nullable=False),
        sa.Column('status', sa.String, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
    )

    migration = _load_migration_module()
    with engine.begin() as connection:
        metadata.create_all(connection)
        connection.execute(
            jira_dc_users.insert(),
            [
                {
                    'id': 1,
                    'keycloak_user_id': 'user-1',
                    'jira_dc_user_id': 'jira-user',
                    'jira_dc_workspace_id': 1,
                    'status': 'active',
                    'created_at': datetime(2026, 5, 21),
                    'updated_at': datetime(2026, 5, 21),
                },
                {
                    'id': 2,
                    'keycloak_user_id': 'user-1',
                    'jira_dc_user_id': 'jira-user',
                    'jira_dc_workspace_id': 2,
                    'status': 'active',
                    'created_at': datetime(2026, 5, 22),
                    'updated_at': datetime(2026, 5, 22),
                },
                {
                    'id': 3,
                    'keycloak_user_id': 'user-2',
                    'jira_dc_user_id': 'jira-user-2',
                    'jira_dc_workspace_id': 3,
                    'status': 'active',
                    'created_at': datetime(2026, 5, 21),
                    'updated_at': datetime(2026, 5, 21),
                },
            ],
        )

        context = MigrationContext.configure(connection)
        operations = Operations(context)
        with patch.object(migration, 'op', operations):
            migration.upgrade()

        rows = connection.execute(
            sa.text(
                """
                SELECT id, keycloak_user_id, status
                FROM jira_dc_users
                ORDER BY id
                """
            )
        ).all()
        assert rows == [
            (1, 'user-1', 'inactive'),
            (2, 'user-1', 'active'),
            (3, 'user-2', 'active'),
        ]

        with pytest.raises(IntegrityError):
            connection.execute(
                jira_dc_users.insert(),
                {
                    'id': 4,
                    'keycloak_user_id': 'user-1',
                    'jira_dc_user_id': 'jira-user',
                    'jira_dc_workspace_id': 4,
                    'status': 'active',
                    'created_at': datetime(2026, 5, 23),
                    'updated_at': datetime(2026, 5, 23),
                },
            )


@pytest.mark.asyncio
async def test_update_user_integration_status_targets_workspace_link():
    store = JiraDcIntegrationStore()
    user = Mock(spec=JiraDcUser)
    user.status = 'inactive'

    result = Mock()
    result.scalar_one_or_none.return_value = user
    session = Mock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()

    @asynccontextmanager
    async def mock_session_maker():
        yield session

    with patch('storage.jira_dc_integration_store.a_session_maker', mock_session_maker):
        updated_user = await store.update_user_integration_status(
            'user-1', 42, 'active'
        )

    assert updated_user is user
    assert user.status == 'active'
    session.execute.assert_called_once()
    executed_statement = session.execute.call_args.args[0]
    assert 'jira_dc_users.jira_dc_workspace_id' in str(executed_statement)
    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(user)


@pytest.mark.asyncio
async def test_deactivate_user_links_except_workspace_targets_stale_active_links():
    store = JiraDcIntegrationStore()

    result = Mock()
    result.rowcount = 2
    session = Mock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()

    @asynccontextmanager
    async def mock_session_maker():
        yield session

    with patch('storage.jira_dc_integration_store.a_session_maker', mock_session_maker):
        deactivated_count = await store.deactivate_user_links_except_workspace(
            'user-1', 42
        )

    assert deactivated_count == 2
    session.execute.assert_called_once()
    executed_statement = session.execute.call_args.args[0]
    statement_text = str(executed_statement)
    assert 'jira_dc_users.keycloak_user_id' in statement_text
    assert 'jira_dc_users.jira_dc_workspace_id !=' in statement_text
    assert 'jira_dc_users.status' in statement_text
    session.commit.assert_awaited_once()
