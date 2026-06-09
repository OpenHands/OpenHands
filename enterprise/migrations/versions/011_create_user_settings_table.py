"""create user settings table

Revision ID: 011
Revises: 010
Create Date: 2024-03-11 23:39:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '011'
down_revision: Union[str, None] = '010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'user_settings',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('keycloak_user_id', sa.String(), nullable=True),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('agent', sa.String(), nullable=True),
        sa.Column('max_iterations', sa.Integer(), nullable=True),
        sa.Column('security_analyzer', sa.String(), nullable=True),
        sa.Column('confirmation_mode', sa.Boolean(), nullable=True, default=False),
        sa.Column('llm_model', sa.String(), nullable=True),
        sa.Column('llm_api_key', sa.String(), nullable=True),
        sa.Column('llm_base_url', sa.String(), nullable=True),
        sa.Column('remote_runtime_resource_factor', sa.Integer(), nullable=True),
        sa.Column(
            'enable_default_condenser', sa.Boolean(), nullable=False, default=False
        ),
        sa.Column('user_consents_to_analytics', sa.Boolean(), nullable=True),
        sa.Column('billing_margin', sa.Float(), nullable=True),
        sa.Column(
            'enable_sound_notifications', sa.Boolean(), nullable=True, default=False
        ),
    )
    # Create indexes for faster lookups
    op.create_index('idx_keycloak_user_id', 'user_settings', ['keycloak_user_id'])


def downgrade() -> None:
    op.drop_index('idx_keycloak_user_id', 'user_settings')
    op.drop_table('user_settings')
