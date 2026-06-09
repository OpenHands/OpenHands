"""create saas conversations table

Revision ID: 003
Revises: 002
Create Date: 2025-01-29 09:36:49.475467

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'conversation_metadata',
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False, index=True),
        sa.Column('selected_repository', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('last_updated_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, index=True),
        sa.PrimaryKeyConstraint('conversation_id'),
    )


def downgrade() -> None:
    op.drop_table('conversation_metadata')
