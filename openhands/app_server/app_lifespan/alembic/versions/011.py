"""Add is_paused column to v1_remote_sandbox table

Revision ID: 011
Revises: 010
Create Date: 2026-06-02 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '011'
down_revision: str | None = '010'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add is_paused column to v1_remote_sandbox.

    Enables the concurrency-limit check to query the DB directly by
    (created_by_user_id, is_paused) instead of calling the runtime
    API's global /list endpoint on every conversation start.

    Existing rows default to False (not paused), which is the safe
    assumption: rows without an API-confirmed paused state are treated
    as running, matching the previous behaviour.
    """
    with op.batch_alter_table('v1_remote_sandbox') as batch_op:
        batch_op.add_column(
            sa.Column(
                'is_paused',
                sa.Boolean(),
                nullable=False,
                server_default='0',
            )
        )


def downgrade() -> None:
    with op.batch_alter_table('v1_remote_sandbox') as batch_op:
        batch_op.drop_column('is_paused')
