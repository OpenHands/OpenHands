"""add enable_sound_notifications column to settings table

Revision ID: 007
Revises: 006
Create Date: 2025-05-01 10:00:00.000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'settings',
        sa.Column(
            'enable_sound_notifications', sa.Boolean(), nullable=True, default=False
        ),
    )


def downgrade() -> None:
    op.drop_column('settings', 'enable_sound_notifications')
