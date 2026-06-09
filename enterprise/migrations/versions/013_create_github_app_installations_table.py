"""create user settings table

Revision ID: 013
Revises: 012
Create Date: 2024-03-12 23:39:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '013'
down_revision: Union[str, None] = '012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'github_app_installations',
        sa.Column('id', sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column('installation_id', sa.String(), nullable=False),
        sa.Column('encrypted_token', sa.String(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            onupdate=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            onupdate=sa.text('now()'),
            nullable=False,
        ),
    )
    # Create indexes for faster lookups
    op.create_index(
        'idx_installation_id',
        'github_app_installations',
        ['installation_id'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('idx_installation_id', 'github_app_installations')
    op.drop_table('github_app_installations')
