"""Add org budget configuration tables.

Revision ID: 133
Revises: 132
Create Date: 2026-06-16 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '133'
down_revision: Union[str, None] = '132'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'org_budget_settings',
        sa.Column('id', sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column('org_id', sa.Uuid(), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('monthly_limit', sa.Float(), nullable=True),
        sa.Column('reset_day', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('default_user_monthly_limit', sa.Float(), nullable=True),
        sa.Column('slack_channel', sa.String(), nullable=True),
        sa.Column('slack_team_id', sa.String(), nullable=True),
        sa.Column('cycle_start_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('cycle_start_spend', sa.Float(), nullable=False, server_default='0'),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
        ),
        sa.ForeignKeyConstraint(['org_id'], ['org.id']),
    )
    op.create_index(
        'ix_org_budget_settings_org_id',
        'org_budget_settings',
        ['org_id'],
        unique=True,
    )

    op.create_table(
        'org_budget_threshold',
        sa.Column('id', sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column('org_id', sa.Uuid(), nullable=False),
        sa.Column('percentage', sa.Integer(), nullable=False),
        sa.Column(
            'email_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            'slack_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column('last_triggered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'last_triggered_cycle_start',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
        ),
        sa.ForeignKeyConstraint(['org_id'], ['org.id']),
    )
    op.create_index(
        'ix_org_budget_threshold_org_id',
        'org_budget_threshold',
        ['org_id'],
        unique=False,
    )

    op.create_table(
        'org_user_budget_override',
        sa.Column('id', sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column('org_id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('monthly_limit', sa.Float(), nullable=True),
        sa.Column(
            'is_disabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
        ),
        sa.ForeignKeyConstraint(['org_id'], ['org.id']),
        sa.ForeignKeyConstraint(['user_id'], ['user.id']),
    )
    op.create_index(
        'ix_org_user_budget_override_org_id',
        'org_user_budget_override',
        ['org_id'],
        unique=False,
    )
    op.create_index(
        'ix_org_user_budget_override_user_id',
        'org_user_budget_override',
        ['user_id'],
        unique=False,
    )
    op.create_unique_constraint(
        'uq_org_user_budget_override_org_id_user_id',
        'org_user_budget_override',
        ['org_id', 'user_id'],
    )


def downgrade() -> None:
    op.drop_constraint(
        'uq_org_user_budget_override_org_id_user_id',
        'org_user_budget_override',
        type_='unique',
    )
    op.drop_index(
        'ix_org_user_budget_override_user_id',
        table_name='org_user_budget_override',
    )
    op.drop_index(
        'ix_org_user_budget_override_org_id',
        table_name='org_user_budget_override',
    )
    op.drop_table('org_user_budget_override')

    op.drop_index(
        'ix_org_budget_threshold_org_id',
        table_name='org_budget_threshold',
    )
    op.drop_table('org_budget_threshold')

    op.drop_index(
        'ix_org_budget_settings_org_id',
        table_name='org_budget_settings',
    )
    op.drop_table('org_budget_settings')
