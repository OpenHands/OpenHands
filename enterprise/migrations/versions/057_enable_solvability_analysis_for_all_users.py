"""Enable solvability analysis for all users

Revision ID: 057
Revises: 056
Create Date: 2025-07-15

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '057'
down_revision: Union[str, None] = '056'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update existing rows to True and set default to True
    op.execute('UPDATE user_settings SET enable_solvability_analysis = true')

    # Alter the default value for future rows
    op.alter_column(
        'user_settings',
        'enable_solvability_analysis',
        existing_type=sa.Boolean(),
        server_default=sa.true(),
        existing_nullable=True,
    )


def downgrade() -> None:
    # Revert the default value back to False
    op.alter_column(
        'user_settings',
        'enable_solvability_analysis',
        existing_type=sa.Boolean(),
        server_default=sa.false(),
        existing_nullable=True,
    )

    # Note: We don't revert the data changes in the downgrade function
    # as it would be arbitrary which rows to change back
