"""bump condenser defaults: max_size 120->240, add keep_first column with default 4

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

    1. Update existing users with condenser_max_size=120 to 240
    2. Add condenser_keep_first column
    3. Set condenser_keep_first=4 for existing users who had the old defaults
    """
    # Update existing rows where condenser_max_size is 120 (old default) to 240
    user_settings_table = table(
        'user_settings',
        column('condenser_max_size', sa.Integer),
        column('condenser_keep_first', sa.Integer),
    )
    op.execute(
        user_settings_table.update()
        .where(user_settings_table.c.condenser_max_size == 120)
        .values(condenser_max_size=240)
    )

    # Add condenser_keep_first column
    op.add_column(
        'user_settings',
        sa.Column('condenser_keep_first', sa.Integer(), nullable=True),
    )

    # Set condenser_keep_first=4 for users who had the old SDK default (keep_first=2)
    # Since this column didn't exist before, all existing users were using SDK default
    # We set it to 4 for users who have a condenser_max_size set (meaning they're using condenser)
    op.execute(
        user_settings_table.update()
        .where(user_settings_table.c.condenser_max_size.isnot(None))
        .values(condenser_keep_first=4)
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove condenser_keep_first column
    op.drop_column('user_settings', 'condenser_keep_first')

    # Revert condenser_max_size from 240 back to 120 for affected users
    user_settings_table = table(
        'user_settings', column('condenser_max_size', sa.Integer)
    )
    op.execute(
        user_settings_table.update()
        .where(user_settings_table.c.condenser_max_size == 240)
        .values(condenser_max_size=120)
    )
