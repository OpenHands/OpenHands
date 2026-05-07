"""Add stay_logged_in column to auth_tokens table.

This column allows users to stay logged in with their OAuth providers
(GitHub, GitLab, Bitbucket) across sessions.
"""

from alembic import op
import sqlalchemy as sa


revision = '113_add_stay_logged_in_to_auth_tokens'
down_revision = '112_create_bitbucket_dc_webhook_table'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'auth_tokens',
        sa.Column(
            'stay_logged_in',
            sa.Boolean(),
            nullable=False,
            server_default='false',
        ),
    )


def downgrade() -> None:
    op.drop_column('auth_tokens', 'stay_logged_in')