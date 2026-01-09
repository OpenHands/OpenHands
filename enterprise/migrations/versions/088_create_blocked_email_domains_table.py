"""create blocked_email_domains table

Revision ID: 088
Revises: 087
Create Date: 2025-01-27 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '088'
down_revision: Union[str, None] = '087'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create blocked_email_domains table for storing blocked email domain patterns."""
    op.create_table(
        'blocked_email_domains',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('domain', sa.String(), nullable=False),
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
        sa.PrimaryKeyConstraint('id'),
    )

    # Create unique index on domain column
    op.create_index(
        'ix_blocked_email_domains_domain',
        'blocked_email_domains',
        ['domain'],
        unique=True,
    )


def downgrade() -> None:
    """Drop blocked_email_domains table."""
    op.drop_index('ix_blocked_email_domains_domain', table_name='blocked_email_domains')
    op.drop_table('blocked_email_domains')
