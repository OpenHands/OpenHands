"""Create user_authorizations table and migrate blocked_email_domains

Revision ID: 099
Revises: 098
Create Date: 2025-03-05 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '099'
down_revision: Union[str, None] = '098'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create user_authorizations table, migrate data, and drop blocked_email_domains."""
    # Create user_authorizations table
    op.create_table(
        'user_authorizations',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('email_pattern', sa.String(), nullable=True),
        sa.Column('provider_type', sa.String(), nullable=True),
        sa.Column('type', sa.String(), nullable=False),
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

    # Create index on email_pattern for efficient LIKE queries
    op.create_index(
        'ix_user_authorizations_email_pattern',
        'user_authorizations',
        ['email_pattern'],
    )

    # Create index on type for efficient filtering
    op.create_index(
        'ix_user_authorizations_type',
        'user_authorizations',
        ['type'],
    )

    # Migrate existing blocked_email_domains to user_authorizations as blacklist entries
    # The domain patterns are converted to SQL LIKE patterns:
    # - 'example.com' becomes '%@example.com' (matches user@example.com)
    # - '.us' becomes '%@%.us' (matches user@anything.us)
    # We also add '%.' prefix for subdomain matching
    op.execute("""
        INSERT INTO user_authorizations (email_pattern, provider_type, type, created_at, updated_at)
        SELECT
            CASE
                WHEN domain LIKE '.%' THEN '%' || domain
                ELSE '%@%' || domain
            END as email_pattern,
            NULL as provider_type,
            'blacklist' as type,
            created_at,
            updated_at
        FROM blocked_email_domains
    """)

    # Drop blocked_email_domains table
    op.drop_index('ix_blocked_email_domains_domain', table_name='blocked_email_domains')
    op.drop_table('blocked_email_domains')


def downgrade() -> None:
    """Recreate blocked_email_domains table and migrate data back."""
    # Recreate blocked_email_domains table
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

    op.create_index(
        'ix_blocked_email_domains_domain',
        'blocked_email_domains',
        ['domain'],
        unique=True,
    )

    # Migrate blacklist entries back to blocked_email_domains
    # Reverse the pattern transformation
    op.execute("""
        INSERT INTO blocked_email_domains (domain, created_at, updated_at)
        SELECT
            CASE
                WHEN email_pattern LIKE '%@%.' THEN SUBSTRING(email_pattern FROM 4)
                ELSE SUBSTRING(email_pattern FROM 2)
            END as domain,
            created_at,
            updated_at
        FROM user_authorizations
        WHERE type = 'blacklist' AND provider_type IS NULL
    """)

    # Drop user_authorizations table
    op.drop_index('ix_user_authorizations_type', table_name='user_authorizations')
    op.drop_index('ix_user_authorizations_email_pattern', table_name='user_authorizations')
    op.drop_table('user_authorizations')
