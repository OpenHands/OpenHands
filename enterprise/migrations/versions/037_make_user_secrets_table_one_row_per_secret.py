"""make user secrets table one row per secret

Revision ID: 037
Revises: 036
Create Date: 2024-03-11 23:39:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '037'
down_revision: Union[str, None] = '036'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the old custom_secrets column
    op.drop_column('user_secrets', 'custom_secrets')

    # Add new columns for secret name, value, and description
    op.add_column('user_secrets', sa.Column('secret_name', sa.String(), nullable=False))
    op.add_column(
        'user_secrets', sa.Column('secret_value', sa.String(), nullable=False)
    )
    op.add_column('user_secrets', sa.Column('description', sa.String(), nullable=True))


def downgrade() -> None:
    # Drop the new columns added in the upgrade
    op.drop_column('user_secrets', 'secret_name')
    op.drop_column('user_secrets', 'secret_value')
    op.drop_column('user_secrets', 'description')

    # Re-add the custom_secrets column
    op.add_column('user_secrets', sa.Column('custom_secrets', sa.JSON(), nullable=True))
