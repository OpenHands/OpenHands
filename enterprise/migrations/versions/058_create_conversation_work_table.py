"""create conversation_work table

Revision ID: 058
Revises: 057
Create Date: 2025-07-11 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '058'
down_revision: Union[str, None] = '057'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'conversation_work',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('seconds', sa.Float(), nullable=False, default=0.0),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('conversation_id'),
    )

    # Create indexes
    op.create_index(
        'ix_conversation_work_conversation_id', 'conversation_work', ['conversation_id']
    )
    op.create_index('ix_conversation_work_user_id', 'conversation_work', ['user_id'])
    op.create_index(
        'ix_conversation_work_user_conversation',
        'conversation_work',
        ['user_id', 'conversation_id'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_conversation_work_user_conversation', table_name='conversation_work'
    )
    op.drop_index('ix_conversation_work_user_id', table_name='conversation_work')
    op.drop_index(
        'ix_conversation_work_conversation_id', table_name='conversation_work'
    )
    op.drop_table('conversation_work')
