"""add proactive conversation starters column

Revision ID: 030
Revises: 029
Create Date: 2025-04-30

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '030'
down_revision = '029'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'user_settings',
        sa.Column(
            'enable_proactive_conversation_starters',
            sa.Boolean(),
            nullable=False,
            default=True,
            server_default='TRUE',
        ),
    )


def downgrade():
    op.drop_column('user_settings', 'enable_proactive_conversation_starters')
