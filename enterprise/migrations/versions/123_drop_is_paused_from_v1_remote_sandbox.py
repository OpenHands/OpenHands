"""Drop is_paused from v1_remote_sandbox.

The column was added in 121 to cache runtime pause state in the DB, but it is
not a reliable indicator of whether a sandbox is actually running: only
explicit app-level pause/resume calls update it, so externally terminated or
expired sandboxes remain is_paused=False forever. Concurrency counting now
uses the runtime /list endpoint as the source of truth instead.

Revision ID: 123
Revises: 122
Create Date: 2026-06-05
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '123'
down_revision: Union[str, None] = '122'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('v1_remote_sandbox', 'is_paused')


def downgrade() -> None:
    op.add_column(
        'v1_remote_sandbox',
        sa.Column(
            'is_paused',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
