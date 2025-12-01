"""Add v1_enabled column to user_settings

Revision ID: 083
Revises: 082
Create Date: 2025-11-18 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '083'
down_revision: Union[str, None] = '082'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add v1_enabled column to user_settings table."""
    op.add_column(
        'user_settings',
        sa.Column(
            'v1_enabled',
            sa.Boolean(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove v1_enabled column from user_settings table."""
    op.drop_column('user_settings', 'v1_enabled')
