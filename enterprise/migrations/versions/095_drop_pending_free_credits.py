"""Drop pending_free_credits column from org table.

Revision ID: 095
Revises: 094
Create Date: 2025-02-18 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '095'
down_revision: Union[str, None] = '094'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the pending_free_credits column from org table.
    # This column was used for tracking free credit eligibility but is no longer needed.
    op.drop_column('org', 'pending_free_credits')


def downgrade() -> None:
    # Re-add pending_free_credits column with default false.
    op.add_column(
        'org',
        sa.Column(
            'pending_free_credits',
            sa.Boolean,
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
