"""Create user_authorizations table and migrate blocked_email_domains

Revision ID: 099
Revises: 098
Create Date: 2025-03-05 00:00:00.000000

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '099'
down_revision: Union[str, None] = '098'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _seed_from_environment() -> None:
    """Seed user_authorizations table from environment variables.

    Reads EMAIL_PATTERN_BLACKLIST and EMAIL_PATTERN_WHITELIST environment variables.
    Each should be a comma-separated list of SQL LIKE patterns (e.g., '%@example.com').

    If the environment variables are not set or empty, this function does nothing.

    This allows us to set up feature deployments with particular patterns already
    blacklisted or whitelisted. (For example, you could blacklist everything with
    `%`, and then whitelist certain email accounts.)
    """
    blacklist_patterns = os.environ.get('EMAIL_PATTERN_BLACKLIST', '').strip()
    whitelist_patterns = os.environ.get('EMAIL_PATTERN_WHITELIST', '').strip()

    connection = op.get_bind()

    if blacklist_patterns:
        for pattern in blacklist_patterns.split(','):
            pattern = pattern.strip()
            if pattern:
                connection.execute(
                    sa.text("""
                        INSERT INTO user_authorizations
                            (email_pattern, provider_type, type)
                        VALUES
                            (:pattern, NULL, 'blacklist')
                    """),
                    {'pattern': pattern},
                )

    if whitelist_patterns:
        for pattern in whitelist_patterns.split(','):
            pattern = pattern.strip()
            if pattern:
                connection.execute(
                    sa.text("""
                        INSERT INTO user_authorizations
                            (email_pattern, provider_type, type)
                        VALUES
                            (:pattern, NULL, 'whitelist')
                    """),
                    {'pattern': pattern},
                )


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

    # Seed additional patterns from environment variables (if set)
    _seed_from_environment()


def downgrade() -> None:
    """Recreate blocked_email_domains table and migrate data back."""
    # Drop user_authorizations table
    op.drop_index('ix_user_authorizations_type', table_name='user_authorizations')
    op.drop_index(
        'ix_user_authorizations_email_pattern', table_name='user_authorizations'
    )
    op.drop_table('user_authorizations')
