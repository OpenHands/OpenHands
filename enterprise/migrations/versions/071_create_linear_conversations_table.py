"""create linear_conversations table

Revision ID: 071
Revises: 070
Create Date: 2025-07-08 10:08:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '071'
down_revision: Union[str, None] = '070'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'linear_conversations',
        sa.Column(
            'id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True
        ),
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('issue_id', sa.String(), nullable=False),
        sa.Column('issue_key', sa.String(), nullable=False),
        sa.Column('parent_id', sa.String(), nullable=True),
        sa.Column('linear_user_id', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
    )

    # Create indexes
    op.create_index(
        'ix_linear_conversations_conversation_id',
        'linear_conversations',
        ['conversation_id'],
    )
    op.create_index(
        'ix_linear_conversations_issue_id', 'linear_conversations', ['issue_id']
    )
    op.create_index(
        'ix_linear_conversations_issue_key', 'linear_conversations', ['issue_key']
    )
    op.create_index(
        'ix_linear_conversations_linear_user_id',
        'linear_conversations',
        ['linear_user_id'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_linear_conversations_linear_user_id', table_name='linear_conversations'
    )
    op.drop_index(
        'ix_linear_conversations_issue_key', table_name='linear_conversations'
    )
    op.drop_index('ix_linear_conversations_issue_id', table_name='linear_conversations')
    op.drop_index(
        'ix_linear_conversations_conversation_id', table_name='linear_conversations'
    )
    op.drop_table('linear_conversations')
