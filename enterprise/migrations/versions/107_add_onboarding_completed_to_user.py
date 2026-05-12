"""Add onboarding_completed column to user table.

Tracks whether a user has completed the onboarding flow.
Used to redirect new SaaS users to /onboarding after accepting TOS.

Revision ID: 107
Revises: 106
Create Date: 2026-03-31
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '107'
down_revision: Union[str, None] = '106'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user',
        sa.Column('onboarding_completed', sa.Boolean(), nullable=True, default=False),
    )


def downgrade() -> None:
    op.drop_column('user', 'onboarding_completed')
