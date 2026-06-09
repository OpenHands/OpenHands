"""update enable_default_condenser default to True

Revision ID: 024
Revises: 023
Create Date: 2024-04-08 15:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.sql import column, table

# revision identifiers, used by Alembic.
revision: str = '024'
down_revision: Union[str, None] = '023'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update existing rows in settings table
    settings_table = table('settings', column('enable_default_condenser', sa.Boolean))
    op.execute(settings_table.update().values(enable_default_condenser=True))

    # Update existing rows in user_settings table
    user_settings_table = table(
        'user_settings', column('enable_default_condenser', sa.Boolean)
    )
    op.execute(user_settings_table.update().values(enable_default_condenser=True))

    # Alter the default value for settings table
    op.alter_column(
        'settings',
        'enable_default_condenser',
        existing_type=sa.Boolean(),
        server_default=sa.true(),
        existing_nullable=False,
    )

    # Alter the default value for user_settings table
    op.alter_column(
        'user_settings',
        'enable_default_condenser',
        existing_type=sa.Boolean(),
        server_default=sa.true(),
        existing_nullable=False,
    )


def downgrade() -> None:
    # Revert the default value for settings table
    op.alter_column(
        'settings',
        'enable_default_condenser',
        existing_type=sa.Boolean(),
        server_default=sa.false(),
        existing_nullable=False,
    )

    # Revert the default value for user_settings table
    op.alter_column(
        'user_settings',
        'enable_default_condenser',
        existing_type=sa.Boolean(),
        server_default=sa.false(),
        existing_nullable=False,
    )

    # Note: We don't revert the data changes in the downgrade function
    # as it would be arbitrary which rows to change back
