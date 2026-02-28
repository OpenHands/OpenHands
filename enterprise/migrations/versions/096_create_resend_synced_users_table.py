"""Create resend_synced_users table.

Revision ID: 096
Revises: 095
Create Date: 2025-02-17 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '096'
down_revision: Union[str, None] = '095'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create resend_synced_users table for tracking users synced to Resend audiences."""
    op.create_table(
        'resend_synced_users',
        sa.Column(
            'id',
            sa.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('audience_id', sa.String(), nullable=False),
        sa.Column(
            'synced_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
        ),
        sa.Column('keycloak_user_id', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'email', 'audience_id', name='uq_resend_synced_email_audience'
        ),
    )

    # Create index on email for fast lookups
    op.create_index(
        'ix_resend_synced_users_email',
        'resend_synced_users',
        ['email'],
    )

    # Create index on audience_id for filtering by audience
    op.create_index(
        'ix_resend_synced_users_audience_id',
        'resend_synced_users',
        ['audience_id'],
    )


def downgrade() -> None:
    """Drop resend_synced_users table."""
    op.drop_index(
        'ix_resend_synced_users_audience_id', table_name='resend_synced_users'
    )
    op.drop_index('ix_resend_synced_users_email', table_name='resend_synced_users')
    op.drop_table('resend_synced_users')
