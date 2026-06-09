"""add branch column to convo metadata table

Revision ID: 026
Revises: 025
Create Date: 2025-04-16 14:59:09.415

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '026'
down_revision: Union[str, None] = '025'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'conversation_metadata',
        sa.Column('trigger', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('conversation_metadata', 'trigger')
