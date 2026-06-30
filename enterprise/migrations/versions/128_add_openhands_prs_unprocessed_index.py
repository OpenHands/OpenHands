"""Add composite index on openhands_prs for the unprocessed-PR worker query

get_unprocessed_prs filters on (provider, processed) and orders by updated_at
DESC, but only repo_id / pr_number / status were indexed. Every enrichment tick
therefore full-scanned the table — openhands_prs was the 4th-heaviest
sequential-scan table in prod (~72K seq scans / 13.9B rows read on ~648K live
rows, INC-95). This composite index covers that query path.

Plain CREATE INDEX (not CONCURRENTLY): the enterprise migration harness runs
inside a transaction with a session advisory lock, where alembic's
autocommit_block() (required for CREATE INDEX CONCURRENTLY) raises — see
migration 117. The single-table build is brief and the short write lock is
acceptable, consistent with migrations 117 and 118.

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
        'ix_openhands_prs_provider_processed_updated_at',
        'openhands_prs',
        ['provider', 'processed', 'updated_at'],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        'ix_openhands_prs_provider_processed_updated_at',
        table_name='openhands_prs',
        if_exists=True,
    )
