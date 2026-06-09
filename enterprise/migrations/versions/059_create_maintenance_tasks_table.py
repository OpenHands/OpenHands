"""Create maintenance tasks table

Revision ID: 059
Revises: 058
Create Date: 2025-07-19

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = '059'
down_revision = '058'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'maintenance_tasks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'INACTIVE',
                'PENDING',
                'WORKING',
                'COMPLETED',
                'ERROR',
                name='maintenancetaskstatus',
            ),
            default='INACTIVE',
            nullable=False,
            index=True,
        ),
        sa.Column('processor_type', sa.String(), nullable=False),
        sa.Column('processor_json', sa.Text(), nullable=False),
        sa.Column('delay', sa.Integer(), nullable=False, default=0),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('info', JSON, nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade():
    op.drop_table('maintenance_tasks')
    op.execute('DROP TYPE maintenancetaskstatus')
