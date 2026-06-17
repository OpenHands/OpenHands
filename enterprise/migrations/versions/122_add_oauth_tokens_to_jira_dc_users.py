"""Add OAuth token columns to jira_dc_users.

Revision ID: 122
Revises: 121
Create Date: 2025-05-31
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '122'
down_revision: Union[str, None] = '121'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = 'jira_dc_users'


def _token_columns() -> tuple[sa.Column, ...]:
    return (
        sa.Column('oauth_access_token_encrypted', sa.String(), nullable=True),
        sa.Column('oauth_refresh_token_encrypted', sa.String(), nullable=True),
        # Epoch seconds; 0 = unknown / no expiry info supplied by the IdP.
        sa.Column('oauth_access_token_expires_at', sa.BigInteger(), nullable=True),
        sa.Column('oauth_refresh_token_expires_at', sa.BigInteger(), nullable=True),
    )


def _existing_columns() -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {column['name'] for column in inspector.get_columns(TABLE_NAME)}


def upgrade() -> None:
    existing = _existing_columns()
    for column in _token_columns():
        if column.name not in existing:
            op.add_column(TABLE_NAME, column)


def downgrade() -> None:
    existing = _existing_columns()
    for column in reversed(_token_columns()):
        if column.name in existing:
            op.drop_column(TABLE_NAME, column.name)
