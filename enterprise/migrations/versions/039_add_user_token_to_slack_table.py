"""add user token to conversation metadata table

Revision ID: 039
Revises: 038
Create Date: 2025-05-16 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '039'
down_revision: Union[str, None] = '038'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'slack_users',
        sa.Column('slack_user_token', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('slack_users', 'slack_user_token')
