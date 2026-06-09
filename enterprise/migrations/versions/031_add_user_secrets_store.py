"""create user secrets table

Revision ID: 031
Revises: 030
Create Date: 2024-03-11 23:39:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '031'
down_revision: Union[str, None] = '030'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'user_secrets',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('keycloak_user_id', sa.String(), nullable=True),
        sa.Column('custom_secrets', sa.JSON(), nullable=True),
    )
    # Create indexes for faster lookups
    op.create_index(
        'idx_user_secrets_keycloak_user_id', 'user_secrets', ['keycloak_user_id']
    )


def downgrade() -> None:
    op.drop_index('idx_user_secrets_keycloak_user_id', 'user_secrets')
    op.drop_table('user_secrets')
