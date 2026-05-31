"""Add OAuth token columns to jira_dc_users.

Revision ID: 117
Revises: 116
Create Date: 2025-05-31
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '117'
down_revision: Union[str, None] = '116'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('jira_dc_users') as batch:
        batch.add_column(
            sa.Column('oauth_access_token_encrypted', sa.String(), nullable=True)
        )
        batch.add_column(
            sa.Column('oauth_refresh_token_encrypted', sa.String(), nullable=True)
        )
        # Epoch seconds; 0 = unknown / no expiry info supplied by the IdP.
        batch.add_column(
            sa.Column('oauth_access_token_expires_at', sa.BigInteger(), nullable=True)
        )
        batch.add_column(
            sa.Column('oauth_refresh_token_expires_at', sa.BigInteger(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table('jira_dc_users') as batch:
        for col in (
            'oauth_refresh_token_expires_at',
            'oauth_access_token_expires_at',
            'oauth_refresh_token_encrypted',
            'oauth_access_token_encrypted',
        ):
            batch.drop_column(col)
