"""add parent_id column and index to slack conversation table

Revision ID: 043
Revises: 042
Create Date: 2025-06-03 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '043'
down_revision: Union[str, None] = '042'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add parent_id column
    op.add_column(
        'slack_conversation', sa.Column('parent_id', sa.String(), nullable=True)
    )

    # Create index on parent_id column
    op.create_index(
        'ix_slack_conversation_parent_id', 'slack_conversation', ['parent_id']
    )


def downgrade() -> None:
    # Drop index first
    op.drop_index('ix_slack_conversation_parent_id', table_name='slack_conversation')

    # Then drop column
    op.drop_column('slack_conversation', 'parent_id')
