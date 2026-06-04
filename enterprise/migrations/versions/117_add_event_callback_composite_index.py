"""Add composite index on event_callback for execute_callbacks query

The execute_callbacks query filters on (conversation_id, status, event_kind)
but none of these columns were indexed, causing full sequential scans on the
event_callback table for every event dispatch (INC-95). This composite index
directly covers that query.

The equivalent index is defined for the OSS app_server alembic chain in
openhands/app_server/app_lifespan/alembic/versions/010.py. That chain is only
run by OssAppLifespanService; the SaaS deployment applies DB migrations solely
through this enterprise chain, so the index must also be created here. Both use
IF NOT EXISTS so they are safe to coexist.

CREATE INDEX CONCURRENTLY is used (inside an autocommit block) to avoid locking
the table during deployment.

Revision ID: 117
Revises: 116
Create Date: 2026-06-04
"""

from typing import Sequence, Union

from alembic import op

revision: str = '117'
down_revision: Union[str, None] = '116'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


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
