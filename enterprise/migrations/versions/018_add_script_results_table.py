"""Add a table for tracking output from maintenance scripts. These are basically migrations that are not sql centric.
Revision ID: 018
Revises: 017
Create Date: 2025-03-26 19:45:00.000

"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '018'
down_revision: Union[str, None] = '017'
branch_labels: Union[str, sa.Sequence[str], None] = None
depends_on: Union[str, sa.Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'script_results',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('revision', sa.String(), nullable=False, index=True),
        sa.Column('data', sa.JSON()),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table('script_results')
