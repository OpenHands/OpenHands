"""Add dependency_repos_cloned column to app_conversation_start_task table

Revision ID: 015
Revises: 014
Create Date: 2026-06-26 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '015'
down_revision: str | None = '014'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add dependency_repos_cloned column to app_conversation_start_task.

    Stores the absolute sandbox paths of any dependency repositories cloned
    for the conversation. Existing rows default to an empty list.
    """
    with op.batch_alter_table('app_conversation_start_task') as batch_op:
        batch_op.add_column(
            sa.Column(
                'dependency_repos_cloned',
                sa.JSON(),
                nullable=False,
                server_default='[]',
            )
        )


def downgrade() -> None:
    with op.batch_alter_table('app_conversation_start_task') as batch_op:
        batch_op.drop_column('dependency_repos_cloned')
