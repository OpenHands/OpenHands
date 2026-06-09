"""create auth tokens table

Revision ID: 021
Revises: 020
Create Date: 2025-03-30 20:15:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '021'
down_revision: Union[str, None] = '020'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'auth_tokens',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('keycloak_user_id', sa.String(), nullable=False),
        sa.Column('identity_provider', sa.String(), nullable=False),
        sa.Column('access_token', sa.String(), nullable=False),
        sa.Column('refresh_token', sa.String(), nullable=False),
        sa.Column('access_token_expires_at', sa.BigInteger(), nullable=False),
        sa.Column('refresh_token_expires_at', sa.BigInteger(), nullable=False),
    )
    op.create_index(
        'idx_auth_tokens_keycloak_user_id', 'auth_tokens', ['keycloak_user_id']
    )
    op.create_index(
        'idx_auth_tokens_keycloak_user_identity_provider',
        'auth_tokens',
        ['keycloak_user_id', 'identity_provider'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('idx_auth_tokens_keycloak_user_identity_provider', 'auth_tokens')
    op.drop_index('idx_auth_tokens_keycloak_user_id', 'auth_tokens')
    op.drop_table('auth_tokens')
