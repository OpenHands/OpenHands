"""Add composite index on event_callback for execute_callbacks query

The execute_callbacks query filters on (conversation_id, status, event_kind)
but none of these columns were indexed, causing full sequential scans on the
event_callback table for every event dispatch (INC-95). This composite index
directly covers that query.

Implementation note: this migration runs through the enterprise alembic harness
(migrations/env.py), which acquires a session-level pg_advisory_lock on the
migration connection before opening the migration transaction. That pre-opened
transaction makes alembic's autocommit_block() (required for CREATE INDEX
CONCURRENTLY) raise an AssertionError, so a CONCURRENTLY build is not possible
here. event_callback is small, so a plain transactional CREATE INDEX completes
in well under a second; the brief lock on writes during the build is acceptable
and is consistent with every other migration in this chain.

The OSS app_server chain creates the equivalent index in
openhands/app_server/app_lifespan/alembic/versions/010.py. Both use IF NOT
EXISTS so they are safe to coexist across deployment modes.

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
    op.create_index(
        'ix_event_callback_conversation_id_status_event_kind',
        'event_callback',
        ['conversation_id', 'status', 'event_kind'],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        'ix_event_callback_conversation_id_status_event_kind',
        table_name='event_callback',
        if_exists=True,
    )
