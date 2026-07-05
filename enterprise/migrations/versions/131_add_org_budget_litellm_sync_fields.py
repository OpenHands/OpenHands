"""Add LiteLLM sync metadata to org budgets.

Revision ID: 131
Revises: 130
Create Date: 2026-06-16 12:24:55.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '131'
down_revision: Union[str, None] = '130'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'org_budget_settings',
        sa.Column('litellm_last_sync_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        'org_budget_settings',
        sa.Column('litellm_last_sync_status', sa.String(), nullable=True),
    )
    op.add_column(
        'org_budget_settings',
        sa.Column('litellm_last_sync_error', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('org_budget_settings', 'litellm_last_sync_error')
    op.drop_column('org_budget_settings', 'litellm_last_sync_status')
    op.drop_column('org_budget_settings', 'litellm_last_sync_at')
