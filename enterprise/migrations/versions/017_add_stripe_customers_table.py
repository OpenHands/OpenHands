"""Add a stripe customers table

Revision ID: 017
Revises: 016
Create Date: 2025-03-20 16:30:00.000

"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '017'
down_revision: Union[str, None] = '016'
branch_labels: Union[str, sa.Sequence[str], None] = None
depends_on: Union[str, sa.Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'stripe_customers',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('keycloak_user_id', sa.String(), nullable=False),
        sa.Column('stripe_customer_id', sa.String(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            onupdate=sa.text('now()'),
            nullable=False,
        ),
    )
    # Create indexes for faster lookups
    op.create_index(
        'idx_stripe_customers_keycloak_user_id',
        'stripe_customers',
        ['keycloak_user_id'],
    )
    op.create_index(
        'idx_stripe_customers_stripe_customer_id',
        'stripe_customers',
        ['stripe_customer_id'],
    )


def downgrade() -> None:
    op.drop_table('stripe_customers')
