"""Add agent_kind column to conversation_metadata table.

Stores the agent type ('llm' or 'acp') for each conversation so the
correct agent-server endpoint can be used when routing requests.

Revision ID: 110
Revises: 109
Create Date: 2026-04-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '110'
down_revision: Union[str, None] = '109'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'conversation_metadata',
        sa.Column('agent_kind', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('conversation_metadata', 'agent_kind')
