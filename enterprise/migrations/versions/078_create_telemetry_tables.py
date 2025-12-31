"""create telemetry tables

Revision ID: 078
Revises: 077
Create Date: 2025-10-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '078'
down_revision: Union[str, None] = '077'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create telemetry tables for metrics collection and configuration."""
    # Create telemetry_metrics table
    op.create_table(
        'telemetry_metrics',
        sa.Column(
            'id',
            sa.String(),  # UUID as string
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            'collected_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
        ),
        sa.Column(
            'metrics_data',
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            'uploaded_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            'upload_attempts',
            sa.Integer(),
            nullable=False,
            server_default='0',
        ),
        sa.Column(
            'last_upload_error',
            sa.Text(),
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
    )

    # Create indexes for telemetry_metrics
    op.create_index(
        'ix_telemetry_metrics_collected_at', 'telemetry_metrics', ['collected_at']
    )
    op.create_index(
        'ix_telemetry_metrics_uploaded_at', 'telemetry_metrics', ['uploaded_at']
    )

    # Create telemetry_replicated_identity table (minimal persistent identity data)
    op.create_table(
        'telemetry_replicated_identity',
        sa.Column(
            'id',
            sa.Integer(),
            nullable=False,
            primary_key=True,
            server_default='1',
        ),
        sa.Column(
            'customer_id',
            sa.String(255),
            nullable=True,
        ),
        sa.Column(
            'instance_id',
            sa.String(255),
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
    )

    # Add constraint to ensure single row in telemetry_replicated_identity
    op.create_check_constraint(
        'single_identity_row', 'telemetry_replicated_identity', 'id = 1'
    )


def downgrade() -> None:
    """Drop telemetry tables."""
    # Drop indexes first
    op.drop_index('ix_telemetry_metrics_uploaded_at', 'telemetry_metrics')
    op.drop_index('ix_telemetry_metrics_collected_at', 'telemetry_metrics')

    # Drop tables
    op.drop_table('telemetry_replicated_identity')
    op.drop_table('telemetry_metrics')
