"""Create device_codes table for OAuth 2.0 Device Flow

Revision ID: 084
Revises: 083
Create Date: 2024-12-10 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '084'
down_revision = '083'
branch_labels = None
depends_on = None


def upgrade():
    """Create device_codes table for OAuth 2.0 Device Flow."""
    op.create_table(
        'device_codes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('device_code', sa.String(length=128), nullable=False),
        sa.Column('user_code', sa.String(length=16), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('keycloak_user_id', sa.String(length=255), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('authorized_at', sa.DateTime(timezone=True), nullable=True),
        # Rate limiting fields for RFC 8628 section 3.5 compliance
        sa.Column('last_poll_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('current_interval', sa.Integer(), nullable=False, default=5),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for efficient lookups
    op.create_index(
        'ix_device_codes_device_code', 'device_codes', ['device_code'], unique=True
    )
    op.create_index(
        'ix_device_codes_user_code', 'device_codes', ['user_code'], unique=True
    )


def downgrade():
    """Drop device_codes table."""
    op.drop_index('ix_device_codes_user_code', table_name='device_codes')
    op.drop_index('ix_device_codes_device_code', table_name='device_codes')
    op.drop_table('device_codes')
