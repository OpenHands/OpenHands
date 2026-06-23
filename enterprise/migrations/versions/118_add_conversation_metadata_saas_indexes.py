"""Add indexes on conversation_metadata_saas for user_id / org_id lookups

conversation_metadata_saas only had the conversation_id primary-key index, so
every query filtering by user_id or org_id did a full table scan — the
2nd-heaviest sequential-scan table in prod (~680K seq scans, ~42B rows read
since mid-December).

Indexes added (matched to the actual query paths):
  - (user_id, org_id): the conversation-listing path filters by user_id and
    optionally org_id; the leftmost prefix also serves the user_id-only delete.
  - (org_id): org-admin paths select/delete by org_id alone.

Plain CREATE INDEX (not CONCURRENTLY): the enterprise migration harness runs
inside a transaction with an advisory lock, where alembic's autocommit_block
fails (see migration 117). conversation_metadata_saas is small (~124K rows /
12MB), so the build is sub-second and the brief write lock is acceptable.

Revision ID: 118
Revises: 117
Create Date: 2026-06-04
"""

from typing import Sequence, Union

from alembic import op

revision: str = '118'
down_revision: Union[str, None] = '117'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'ix_conversation_metadata_saas_user_id_org_id',
        'conversation_metadata_saas',
        ['user_id', 'org_id'],
        if_not_exists=True,
    )
    op.create_index(
        'ix_conversation_metadata_saas_org_id',
        'conversation_metadata_saas',
        ['org_id'],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        'ix_conversation_metadata_saas_org_id',
        table_name='conversation_metadata_saas',
        if_exists=True,
    )
    op.drop_index(
        'ix_conversation_metadata_saas_user_id_org_id',
        table_name='conversation_metadata_saas',
        if_exists=True,
    )
