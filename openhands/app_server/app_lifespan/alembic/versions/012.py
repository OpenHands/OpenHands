"""Drop is_paused column from v1_remote_sandbox table

Revision ID: 012
Revises: 011
Create Date: 2026-06-05 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '012'
down_revision: str | None = '011'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the is_paused column.

    The column was added in 011 to cache runtime state in the DB, but it is
    not a reliable indicator of whether a sandbox is actually running: only
    explicit app-level pause/resume calls update it, so externally terminated
    or expired sandboxes stay is_paused=False forever.  Concurrency counting
    now uses the runtime /list endpoint as the source of truth.
    """
    with op.batch_alter_table('v1_remote_sandbox') as batch_op:
        batch_op.drop_column('is_paused')


def downgrade() -> None:
    with op.batch_alter_table('v1_remote_sandbox') as batch_op:
        batch_op.add_column(
            sa.Column(
                'is_paused',
                sa.Boolean(),
                nullable=False,
                server_default='0',
            )
        )
