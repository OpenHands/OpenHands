"""create saas conversations table

Revision ID: 004
Revises: 003
Create Date: 2025-01-29 09:36:49.475467

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'billing_sessions',
        sa.Column('id', sa.String(), nullable=False, primary_key=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'in_progress',
                'completed',
                'cancelled',
                'error',
                name='billing_session_status_enum',
            ),
            nullable=False,
            default='in_progress',
        ),
        sa.Column('price', sa.DECIMAL(19, 4), nullable=False),
        sa.Column('price_code', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('billing_sessions')
    op.execute('DROP TYPE billing_session_status_enum')
