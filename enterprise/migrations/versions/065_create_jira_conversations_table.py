"""create jira_conversations table

Revision ID: 065
Revises: 064
Create Date: 2025-07-08 10:02:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '065'
down_revision: Union[str, None] = '064'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'jira_conversations',
        sa.Column(
            'id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True
        ),
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('issue_id', sa.String(), nullable=False),
        sa.Column('issue_key', sa.String(), nullable=False),
        sa.Column('parent_id', sa.String(), nullable=True),
        sa.Column('jira_user_id', sa.Integer(), nullable=False),
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
        'ix_jira_conversations_conversation_id',
        'jira_conversations',
        ['conversation_id'],
    )
    op.create_index(
        'ix_jira_conversations_issue_id', 'jira_conversations', ['issue_id']
    )
    op.create_index(
        'ix_jira_conversations_issue_key', 'jira_conversations', ['issue_key']
    )
    op.create_index(
        'ix_jira_conversations_jira_user_id',
        'jira_conversations',
        ['jira_user_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_jira_conversations_jira_user_id', table_name='jira_conversations')
    op.drop_index('ix_jira_conversations_issue_key', table_name='jira_conversations')
    op.drop_index('ix_jira_conversations_issue_id', table_name='jira_conversations')
    op.drop_index(
        'ix_jira_conversations_conversation_id', table_name='jira_conversations'
    )
    op.drop_table('jira_conversations')
