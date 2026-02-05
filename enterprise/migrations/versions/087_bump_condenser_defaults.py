"""bump condenser defaults: max_size 120->240

Revision ID: 087
Revises: 086
Create Date: 2026-01-05

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.sql import column, table

# revision identifiers, used by Alembic.
revision: str = '087'
down_revision: Union[str, None] = '086'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Update existing users with condenser_max_size=120 or NULL to 240.
    This covers both users who had the old default (120) explicitly set
    and users who had NULL (which defaulted to 120 in the application code).
    The SDK default for keep_first will be used automatically.
    """
    user_settings_table = table(
        'user_settings',
        column('condenser_max_size', sa.Integer),
    )
    # Update users with explicit 120 value
    op.execute(
        user_settings_table.update()
        .where(user_settings_table.c.condenser_max_size == 120)
        .values(condenser_max_size=240)
    )
    # Update users with NULL value (which defaulted to 120 in application code)
    op.execute(
        user_settings_table.update()
        .where(user_settings_table.c.condenser_max_size.is_(None))
        .values(condenser_max_size=240)
    )


def downgrade() -> None:
    """Downgrade schema.

    Note: This sets all 240 values back to NULL (not 120) since we can't
    distinguish between users who had 120 vs NULL before the upgrade.
    """
    user_settings_table = table(
        'user_settings', column('condenser_max_size', sa.Integer)
    )
    op.execute(
        user_settings_table.update()
        .where(user_settings_table.c.condenser_max_size == 240)
        .values(condenser_max_size=None)
    )
