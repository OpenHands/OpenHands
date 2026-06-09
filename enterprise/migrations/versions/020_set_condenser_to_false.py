"""set condenser to false for all users

Revision ID: 020
Revises: 019
Create Date: 2025-04-02 12:45:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.sql import column, table

# revision identifiers, used by Alembic.
revision: str = '020'
down_revision: Union[str, None] = '019'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Define tables for update operations
    settings_table = table('settings', column('enable_default_condenser', sa.Boolean))

    user_settings_table = table(
        'user_settings', column('enable_default_condenser', sa.Boolean)
    )

    # Update the enable_default_condenser column to False for all users in the settings table
    op.execute(settings_table.update().values(enable_default_condenser=False))

    # Update the enable_default_condenser column to False for all users in the user_settings table
    op.execute(user_settings_table.update().values(enable_default_condenser=False))


def downgrade() -> None:
    # No downgrade operation needed as we're just setting a value
    # and not changing schema structure
    pass
