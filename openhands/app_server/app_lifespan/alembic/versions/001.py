"""Sync DB with Models

Revision ID: 001
Revises:
Create Date: 2025-10-05 11:28:41.772294

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartTaskStatus,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResultStatus,
)

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'app_conversation_start_task',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_by_user_id', sa.String(), nullable=True),
        sa.Column('status', sa.Enum(AppConversationStartTaskStatus), nullable=True),
        sa.Column('detail', sa.String(), nullable=True),
        sa.Column('app_conversation_id', sa.UUID(), nullable=True),
        sa.Column('sandbox_id', sa.String(), nullable=True),
        sa.Column('agent_server_url', sa.String(), nullable=True),
        sa.Column('request', sa.JSON(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'),
            nullable=True,
        ),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_app_conversation_start_task_created_at'),
        'app_conversation_start_task',
        ['created_at'],
        unique=False,
    )
    op.create_index(
        op.f('ix_app_conversation_start_task_created_by_user_id'),
        'app_conversation_start_task',
        ['created_by_user_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_app_conversation_start_task_updated_at'),
        'app_conversation_start_task',
        ['updated_at'],
        unique=False,
    )
    op.create_table(
        'event_callback',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('conversation_id', sa.UUID(), nullable=True),
        sa.Column('processor', sa.JSON(), nullable=True),
        sa.Column('event_kind', sa.String(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_event_callback_created_at'),
        'event_callback',
        ['created_at'],
        unique=False,
    )
    op.create_table(
        'event_callback_result',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('status', sa.Enum(EventCallbackResultStatus), nullable=True),
        sa.Column('event_callback_id', sa.UUID(), nullable=True),
        sa.Column('event_id', sa.UUID(), nullable=True),
        sa.Column('conversation_id', sa.UUID(), nullable=True),
        sa.Column('detail', sa.String(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_event_callback_result_conversation_id'),
        'event_callback_result',
        ['conversation_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_event_callback_result_created_at'),
        'event_callback_result',
        ['created_at'],
        unique=False,
    )
    op.create_index(
        op.f('ix_event_callback_result_event_callback_id'),
        'event_callback_result',
        ['event_callback_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_event_callback_result_event_id'),
        'event_callback_result',
        ['event_id'],
        unique=False,
    )
    op.create_table(
        'v1_remote_sandbox',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_by_user_id', sa.String(), nullable=True),
        sa.Column('sandbox_spec_id', sa.String(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_v1_remote_sandbox_created_at'),
        'v1_remote_sandbox',
        ['created_at'],
        unique=False,
    )
    op.create_index(
        op.f('ix_v1_remote_sandbox_created_by_user_id'),
        'v1_remote_sandbox',
        ['created_by_user_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_v1_remote_sandbox_sandbox_spec_id'),
        'v1_remote_sandbox',
        ['sandbox_spec_id'],
        unique=False,
    )
    op.create_table(
        'conversation_metadata',
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('github_user_id', sa.String(), nullable=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('selected_repository', sa.String(), nullable=True),
        sa.Column('selected_branch', sa.String(), nullable=True),
        sa.Column('git_provider', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('last_updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('trigger', sa.String(), nullable=True),
        sa.Column('pr_number', sa.JSON(), nullable=True),
        sa.Column('accumulated_cost', sa.Float(), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('max_budget_per_task', sa.Float(), nullable=True),
        sa.Column('cache_read_tokens', sa.Integer(), nullable=True),
        sa.Column('cache_write_tokens', sa.Integer(), nullable=True),
        sa.Column('reasoning_tokens', sa.Integer(), nullable=True),
        sa.Column('context_window', sa.Integer(), nullable=True),
        sa.Column('per_turn_token', sa.Integer(), nullable=True),
        sa.Column('llm_model', sa.String(), nullable=True),
        sa.Column('conversation_version', sa.String(), nullable=False),
        sa.Column('sandbox_id', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('conversation_id'),
    )
    op.create_index(
        op.f('ix_conversation_metadata_conversation_version'),
        'conversation_metadata',
        ['conversation_version'],
        unique=False,
    )
    op.create_index(
        op.f('ix_conversation_metadata_sandbox_id'),
        'conversation_metadata',
        ['sandbox_id'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f('ix_conversation_metadata_sandbox_id'), table_name='conversation_metadata'
    )
    op.drop_index(
        op.f('ix_conversation_metadata_conversation_version'),
        table_name='conversation_metadata',
    )
    op.drop_table('conversation_metadata')
    op.drop_index(
        op.f('ix_v1_remote_sandbox_sandbox_spec_id'), table_name='v1_remote_sandbox'
    )
    op.drop_index(
        op.f('ix_v1_remote_sandbox_created_by_user_id'), table_name='v1_remote_sandbox'
    )
    op.drop_index(
        op.f('ix_v1_remote_sandbox_created_at'), table_name='v1_remote_sandbox'
    )
    op.drop_table('v1_remote_sandbox')
    op.drop_index(
        op.f('ix_event_callback_result_event_id'),
        table_name='event_callback_result',
    )
    op.drop_index(
        op.f('ix_event_callback_result_event_callback_id'),
        table_name='event_callback_result',
    )
    op.drop_index(
        op.f('ix_event_callback_result_created_at'),
        table_name='event_callback_result',
    )
    op.drop_index(
        op.f('ix_event_callback_result_conversation_id'),
        table_name='event_callback_result',
    )
    op.drop_table('event_callback_result')
    op.drop_index(op.f('ix_event_callback_created_at'), table_name='event_callback')
    op.drop_table('event_callback')
    op.drop_index(
        op.f('ix_app_conversation_start_task_updated_at'),
        table_name='app_conversation_start_task',
    )
    op.drop_index(
        op.f('ix_app_conversation_start_task_created_by_user_id'),
        table_name='app_conversation_start_task',
    )
    op.drop_index(
        op.f('ix_app_conversation_start_task_created_at'),
        table_name='app_conversation_start_task',
    )
    op.drop_table('app_conversation_start_task')
    op.drop_index(
        op.f('ix_app_conversation_info_updated_at'),
        table_name='app_conversation_info',
    )
    op.drop_index(
        op.f('ix_app_conversation_info_sandbox_id'),
        table_name='app_conversation_info',
    )
    op.drop_index(
        op.f('ix_app_conversation_info_created_by_user_id'),
        table_name='app_conversation_info',
    )
    op.drop_index(
        op.f('ix_app_conversation_info_created_at'),
        table_name='app_conversation_info',
    )
    op.drop_table('app_conversation_info')
