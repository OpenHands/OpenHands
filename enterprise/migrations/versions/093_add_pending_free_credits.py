"""Add pending_free_credits flag to org table.

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
    # Add pending_free_credits column to org table with default false.
    # New orgs will have this set to TRUE at creation time.
    # Existing orgs default to FALSE (not eligible - they already got $10 at signup).
    op.add_column(
        'org',
        sa.Column(
            'pending_free_credits',
            sa.Boolean,
            nullable=False,
            server_default=sa.text('false'),
        ),
    )


def downgrade() -> None:
    op.drop_column('org', 'pending_free_credits')
