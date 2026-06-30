"""Add indexes on gitlab_webhook and billing_sessions for hot lookup queries

INC-95: two enterprise tables were sequentially scanned on hot paths because
their lookup columns were unindexed.

  - gitlab_webhook (no non-PK index): get / update / delete / reset-by-resource
    filter on project_id or group_id (gitlab_webhook_store) — ~77K seq scans /
    395M rows read. Add single-column indexes on project_id and group_id.
  - billing_sessions: the completed-credit lookup filters on (user_id, status)
    (user_store.py) — ~4.6K seq scans / 42M rows read. Add a composite
    (user_id, status) index. The other lookup filters by id, served by the PK.

resend_synced_users (also flagged in #14634) is intentionally left unchanged: it
is already indexed on email / audience_id, and its remaining scans come from
get_synced_emails_for_audience reading ~all rows of a single audience — a
low-selectivity full read an index cannot improve (needs app-level caching).

Plain CREATE INDEX (not CONCURRENTLY): the enterprise migration harness runs
inside a transaction with a session advisory lock where alembic's
autocommit_block() raises — see migration 117. The single-table builds are brief
and the short write lock is acceptable, consistent with migrations 117 and 118.

Revision ID: 128
Revises: 127
Create Date: 2026-06-30
"""

from typing import Sequence, Union

from alembic import op

revision: str = '128'
down_revision: Union[str, None] = '127'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'ix_gitlab_webhook_project_id',
        'gitlab_webhook',
        ['project_id'],
        if_not_exists=True,
    )
    op.create_index(
        'ix_gitlab_webhook_group_id',
        'gitlab_webhook',
        ['group_id'],
        if_not_exists=True,
    )
    op.create_index(
        'ix_billing_sessions_user_id_status',
        'billing_sessions',
        ['user_id', 'status'],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        'ix_billing_sessions_user_id_status',
        table_name='billing_sessions',
        if_exists=True,
    )
    op.drop_index(
        'ix_gitlab_webhook_group_id',
        table_name='gitlab_webhook',
        if_exists=True,
    )
    op.drop_index(
        'ix_gitlab_webhook_project_id',
        table_name='gitlab_webhook',
        if_exists=True,
    )
