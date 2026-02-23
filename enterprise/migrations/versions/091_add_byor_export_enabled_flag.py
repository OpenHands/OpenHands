"""Add byor_export_enabled flag to org table.

Revision ID: 091
Revises: 090
Create Date: 2025-01-15 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '091'
down_revision: Union[str, None] = '090'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add byor_export_enabled column to org table with default false
    op.add_column(
        'org',
        sa.Column(
            'byor_export_enabled',
            sa.Boolean,
            nullable=False,
            server_default=sa.text('false'),
        ),
    )

    # Set byor_export_enabled to true for orgs that have completed billing sessions
    op.execute(
        sa.text("""
            UPDATE org SET byor_export_enabled = TRUE
            WHERE id IN (
                SELECT DISTINCT org_id FROM billing_sessions
                WHERE status = 'completed' AND org_id IS NOT NULL
            )
        """)
    )


def downgrade() -> None:
    op.drop_column('org', 'byor_export_enabled')
