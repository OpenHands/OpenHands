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
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {
        column['name'] for column in inspector.get_columns('conversation_metadata')
    }
    indexes = {
        index['name'] for index in inspector.get_indexes('conversation_metadata')
    }

    if 'execution_status' not in columns:
        with op.batch_alter_table('conversation_metadata') as batch_op:
            batch_op.add_column(
                sa.Column(
                    'execution_status',
                    sa.String(),
                    nullable=True,
                )
            )

    if 'ix_conversation_metadata_execution_status' not in indexes:
        op.create_index(
            'ix_conversation_metadata_execution_status',
            'conversation_metadata',
            ['execution_status'],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {
        column['name'] for column in inspector.get_columns('conversation_metadata')
    }
    indexes = {
        index['name'] for index in inspector.get_indexes('conversation_metadata')
    }

    if 'ix_conversation_metadata_execution_status' in indexes:
        op.drop_index(
            'ix_conversation_metadata_execution_status',
            table_name='conversation_metadata',
        )

    if 'execution_status' in columns:
        with op.batch_alter_table('conversation_metadata') as batch_op:
            batch_op.drop_column('execution_status')
