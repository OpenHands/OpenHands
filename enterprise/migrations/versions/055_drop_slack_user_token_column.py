"""drop slack_user_token column from slack_users table

Revision ID: 055
Revises: 054
Create Date: 2025-07-07 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '055'
down_revision: Union[str, None] = '054'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('slack_users', 'slack_user_token')


def downgrade() -> None:
    op.add_column(
        'slack_users',
        sa.Column('slack_user_token', sa.String(), nullable=True),
    )
