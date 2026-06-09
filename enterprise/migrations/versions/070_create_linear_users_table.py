"""create linear_users table

Revision ID: 070
Revises: 069
Create Date: 2025-07-08 10:06:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '070'
down_revision: Union[str, None] = '069'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'linear_users',
        sa.Column(
            'id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True
        ),
        sa.Column('keycloak_user_id', sa.String(), nullable=False),
        sa.Column('linear_user_id', sa.String(), nullable=False),
        sa.Column('linear_workspace_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
    )

    # Create indexes
    op.create_index(
        'ix_linear_users_keycloak_user_id', 'linear_users', ['keycloak_user_id']
    )
    op.create_index(
        'ix_linear_users_linear_workspace_id',
        'linear_users',
        ['linear_workspace_id'],
    )
    op.create_index(
        'ix_linear_users_linear_user_id', 'linear_users', ['linear_user_id']
    )


def downgrade() -> None:
    op.drop_index('ix_linear_users_linear_user_id', table_name='linear_users')
    op.drop_index('ix_linear_users_linear_workspace_id', table_name='linear_users')
    op.drop_index('ix_linear_users_keycloak_user_id', table_name='linear_users')
    op.drop_table('linear_users')
