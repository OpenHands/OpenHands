"""Add execution_status column to conversation_metadata table.

Stores the current execution status of conversations
(idle, running, paused, finished, error, stuck, deleting) for org-wide
dashboard queries without requiring agent server calls. This mirrors the
OSS app-lifespan migration 013 (introduced in PR #14846); the enterprise
deployment maintains its own migration chain and therefore needs this
parallel migration.

Revision ID: 127
Revises: 126
Create Date: 2026-06-29 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '127'
down_revision: Union[str, None] = '126'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('conversation_metadata') as batch_op:
        batch_op.add_column(
            sa.Column(
                'execution_status',
                sa.String(),
                nullable=True,
            )
        )
        batch_op.create_index(
            'ix_conversation_metadata_execution_status',
            'execution_status',
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table('conversation_metadata') as batch_op:
        batch_op.drop_index('ix_conversation_metadata_execution_status')
        batch_op.drop_column('execution_status')
