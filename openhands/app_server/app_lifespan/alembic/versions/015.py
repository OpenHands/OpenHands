"""Create conversation_cost_events table (OSS port of enterprise 134/136)

The per-event cost-delta write code in
``sql_app_conversation_info_service.py`` (added in commit acc8412ae,
"feat: track per-event cost deltas") inserts rows into
``conversation_cost_events`` on every positive cost delta. That table is
created by the *enterprise* migration tree (revision 134, cascade-fixed in
136), not the OSS tree, so on a self-hosted OSS install the table never exists
and every cost-delta write raises ``no such table: conversation_cost_events``
(hundreds/hour), breaking cost tracking and poisoning the DB session with
``PendingRollbackError``.

The OSS ``StoredConversationCostEvent`` model already declares the
``ondelete='CASCADE'`` foreign key and both indexes, so this migration simply
materialises the model's schema in the OSS migration set, using the final
upstream form (the enterprise 134 schema with the 136 CASCADE foreign key
already applied).

Revision ID: 015
Revises: 014
Create Date: 2026-07-17 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '015'
down_revision: str | None = '014'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create conversation_cost_events table with cascade-delete foreign key."""
    op.create_table(
        'conversation_cost_events',
        sa.Column('id', sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column(
            'conversation_id',
            sa.String(),
            sa.ForeignKey(
                'conversation_metadata.conversation_id',
                ondelete='CASCADE',
            ),
            nullable=False,
        ),
        sa.Column('cost_delta', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        'ix_conversation_cost_events_conversation_id',
        'conversation_cost_events',
        ['conversation_id'],
    )
    op.create_index(
        'ix_conversation_cost_events_occurred_at',
        'conversation_cost_events',
        ['occurred_at'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_conversation_cost_events_occurred_at',
        table_name='conversation_cost_events',
    )
    op.drop_index(
        'ix_conversation_cost_events_conversation_id',
        table_name='conversation_cost_events',
    )
    op.drop_table('conversation_cost_events')
