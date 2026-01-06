"""add v1 column to slack conversation table

Revision ID: 086
Revises: 085
Create Date: 2025-12-02 15:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '086'
down_revision: Union[str, None] = '085'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add v1 column
    op.add_column(
        'slack_conversation', sa.Column('v1_enabled', sa.Boolean(), nullable=True)
    )


def downgrade() -> None:
    # Drop v1 column
    op.drop_column('slack_conversation', 'v1_enabled')
