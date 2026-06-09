"""create slack conversation table

Revision ID: 041
Revises: 040
Create Date: 2025-05-24 02:40:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '041'
down_revision: Union[str, None] = '040'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'slack_conversation',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('conversation_id', sa.String(), nullable=False, index=True),
        sa.Column('channel_id', sa.String(), nullable=False),
        sa.Column('keycloak_user_id', sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('slack_conversation')
