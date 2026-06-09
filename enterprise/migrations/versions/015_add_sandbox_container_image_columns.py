"""add sandbox_base_container_image and sandbox_runtime_container_image columns

Revision ID: 015
Revises: 014
Create Date: 2025-03-19 19:30:00.000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '015'
down_revision: Union[str, None] = '014'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns to settings table
    op.add_column(
        'settings',
        sa.Column('sandbox_base_container_image', sa.String(), nullable=True),
    )
    op.add_column(
        'settings',
        sa.Column('sandbox_runtime_container_image', sa.String(), nullable=True),
    )

    # Add columns to user_settings table
    op.add_column(
        'user_settings',
        sa.Column('sandbox_base_container_image', sa.String(), nullable=True),
    )
    op.add_column(
        'user_settings',
        sa.Column('sandbox_runtime_container_image', sa.String(), nullable=True),
    )


def downgrade() -> None:
    # Drop columns from settings table
    op.drop_column('settings', 'sandbox_base_container_image')
    op.drop_column('settings', 'sandbox_runtime_container_image')

    # Drop columns from user_settings table
    op.drop_column('user_settings', 'sandbox_base_container_image')
    op.drop_column('user_settings', 'sandbox_runtime_container_image')
