"""fix enable_sound_notifications settings to not be nullable

Revision ID: 009
Revises: 008
Create Date: 2025-02-28 18:28:00.000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '009'
down_revision: Union[str, None] = '008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        'UPDATE settings SET enable_sound_notifications=FALSE where enable_sound_notifications IS NULL'
    )
    op.alter_column(
        'settings',
        sa.Column(
            'enable_sound_notifications', sa.Boolean(), nullable=False, default=False
        ),
    )


def downgrade() -> None:
    op.alter_column(
        'settings',
        sa.Column(
            'enable_sound_notifications', sa.Boolean(), nullable=True, default=False
        ),
    )
