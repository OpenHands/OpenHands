"""Create mcp_server_test_run table.

Revision ID: 013
Revises: 012
Create Date: 2026-06-29
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

from openhands.app_server.mcp.mcp_test_models import (
    MCPServerFailureCategory,
    McpServerTestRunStatus,
    McpServerTransport,
)

revision: str = '013'
down_revision: str | Sequence[str] | None = '012'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'mcp_server_test_run',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_by_user_id', sa.String(), nullable=True),
        sa.Column('server_id', sa.String(), nullable=False),
        sa.Column('transport', sa.Enum(McpServerTransport), nullable=False),
        sa.Column('status', sa.Enum(McpServerTestRunStatus), nullable=False),
        sa.Column('category', sa.Enum(MCPServerFailureCategory), nullable=True),
        sa.Column('message', sa.String(), nullable=True),
        sa.Column('tool_count', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('sandbox_id', sa.String(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_mcp_server_test_run_created_at'),
        'mcp_server_test_run',
        ['created_at'],
        unique=False,
    )
    op.create_index(
        op.f('ix_mcp_server_test_run_created_by_user_id'),
        'mcp_server_test_run',
        ['created_by_user_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_mcp_server_test_run_server_id'),
        'mcp_server_test_run',
        ['server_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_mcp_server_test_run_status'),
        'mcp_server_test_run',
        ['status'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_mcp_server_test_run_status'), table_name='mcp_server_test_run'
    )
    op.drop_index(
        op.f('ix_mcp_server_test_run_server_id'), table_name='mcp_server_test_run'
    )
    op.drop_index(
        op.f('ix_mcp_server_test_run_created_by_user_id'),
        table_name='mcp_server_test_run',
    )
    op.drop_index(
        op.f('ix_mcp_server_test_run_created_at'), table_name='mcp_server_test_run'
    )
    op.drop_table('mcp_server_test_run')
