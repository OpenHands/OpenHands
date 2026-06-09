"""create saas settings table

Revision ID: 002
Revises: 001
Create Date: 2025-01-27 20:08:58.360566

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This was created to match the settings object - in future some of these strings should probabyl
    # be replaced with enum types.
    op.create_table(
        'settings',
        sa.Column('id', sa.String(), nullable=False, primary_key=True),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('agent', sa.String(), nullable=True),
        sa.Column('max_iterations', sa.Integer(), nullable=True),
        sa.Column('security_analyzer', sa.String(), nullable=True),
        sa.Column('confirmation_mode', sa.Boolean(), nullable=True, default=False),
        sa.Column('llm_model', sa.String(), nullable=True),
        sa.Column('llm_api_key', sa.String(), nullable=True),
        sa.Column('llm_base_url', sa.String(), nullable=True),
        sa.Column('remote_runtime_resource_factor', sa.Integer(), nullable=True),
        sa.Column('github_token', sa.String(), nullable=True),
        sa.Column(
            'enable_default_condenser', sa.Boolean(), nullable=False, default=False
        ),
        sa.Column('user_consents_to_analytics', sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('settings')
