"""Add free_credits_granted flag to org table.

Revision ID: 093
Revises: 092
Create Date: 2025-02-17 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '093'
down_revision: Union[str, None] = '092'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add free_credits_granted column to org table with default false
    op.add_column(
        'org',
        sa.Column(
            'free_credits_granted',
            sa.Boolean,
            nullable=False,
            server_default=sa.text('false'),
        ),
    )

    # Mark existing orgs with completed billing sessions as already having received free credits
    # (since they got $10 free credits under the old system)
    op.execute(
        sa.text("""
            UPDATE org SET free_credits_granted = TRUE
            WHERE id IN (
                SELECT DISTINCT org_id FROM billing_sessions
                WHERE status = 'completed' AND org_id IS NOT NULL
            )
        """)
    )


def downgrade() -> None:
    op.drop_column('org', 'free_credits_granted')
