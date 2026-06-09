"""create slack team table

Revision ID: 045
Revises: 044
Create Date: 2025-06-06 21:50:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '045'
down_revision: Union[str, None] = '044'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'slack_teams',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('team_id', sa.String(), nullable=False),
        sa.Column('bot_access_token', sa.String(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
    )

    # Create index for team_id
    op.create_index('ix_slack_teams_team_id', 'slack_teams', ['team_id'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_slack_teams_team_id', table_name='slack_teams')
    op.drop_table('slack_teams')
