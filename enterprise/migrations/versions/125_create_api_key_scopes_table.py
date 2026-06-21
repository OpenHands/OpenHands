"""Create api_key_scopes table.

Revision ID: 125
Revises: 124
Create Date: 2026-06-21 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '125'
down_revision: str | None = '124'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'api_key_scopes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('api_key_id', sa.Integer(), nullable=False),
        sa.Column('scope', sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_api_key_scopes_api_key_id'), 'api_key_scopes', ['api_key_id'], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_api_key_scopes_api_key_id'), table_name='api_key_scopes')
    op.drop_table('api_key_scopes')
