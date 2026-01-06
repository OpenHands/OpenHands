"""bump condenser defaults: max_size 120->240

Revision ID: 086
Revises: 085
Create Date: 2026-01-05

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.sql import column, table

# revision identifiers, used by Alembic.
revision: str = '086'
down_revision: Union[str, None] = '085'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Update existing users with condenser_max_size=120 to 240.
    The SDK default for keep_first will be used automatically.
    """
    user_settings_table = table(
        'user_settings',
        column('condenser_max_size', sa.Integer),
    )
    op.execute(
        user_settings_table.update()
        .where(user_settings_table.c.condenser_max_size == 120)
        .values(condenser_max_size=240)
    )


def downgrade() -> None:
    """Downgrade schema."""
    user_settings_table = table(
        'user_settings', column('condenser_max_size', sa.Integer)
    )
    op.execute(
        user_settings_table.update()
        .where(user_settings_table.c.condenser_max_size == 240)
        .values(condenser_max_size=120)
    )
