"""Add composite index on event_callback for execute_callbacks query

Revision ID: 010
Revises: 009
Create Date: 2026-06-03

The execute_callbacks query filters on (status, event_kind, conversation_id)
but none of these columns were indexed, causing full table scans on every
event dispatch. This index directly covers that query.

CREATE INDEX CONCURRENTLY is used to avoid locking the table during deployment.
"""

from typing import Sequence

from alembic import op

revision: str = '010'
down_revision: str | None = '009'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.create_index(
            'ix_event_callback_conversation_id_status_event_kind',
            'event_callback',
            ['conversation_id', 'status', 'event_kind'],
            postgresql_concurrently=True,
            if_not_exists=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            'ix_event_callback_conversation_id_status_event_kind',
            table_name='event_callback',
            postgresql_concurrently=True,
            if_exists=True,
        )
