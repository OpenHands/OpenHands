"""create slack users table

Revision ID: 035
Revises: 034
Create Date: 2025-04-09 13:45:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '035'
down_revision: Union[str, None] = '034'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'slack_users',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('keycloak_user_id', sa.String(), nullable=False, index=True),
        sa.Column('slack_user_id', sa.String(), nullable=False, index=True),
        sa.Column('slack_display_name', sa.String(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table('slack_users')
