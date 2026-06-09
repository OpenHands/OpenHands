"""create subscription_access table

Revision ID: 074
Revises: 073
Create Date: 2025-08-26

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '074'
down_revision: Union[str, None] = '073'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the ENUM type for subscription access status
    subscription_access_status_enum = sa.Enum(
        'ACTIVE', 'DISABLED', name='subscription_access_status_enum'
    )

    # Create the subscription_access table
    op.create_table(
        'subscription_access',
        sa.Column(
            'id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True
        ),
        sa.Column(
            'status',
            subscription_access_status_enum,
            nullable=False,
        ),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('start_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('amount_paid', sa.DECIMAL(19, 4), nullable=True),
        sa.Column('stripe_invoice_payment_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )

    # Create indexes
    op.create_index('ix_subscription_access_status', 'subscription_access', ['status'])
    op.create_index(
        'ix_subscription_access_user_id', 'subscription_access', ['user_id']
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_subscription_access_user_id', 'subscription_access')
    op.drop_index('ix_subscription_access_status', 'subscription_access')

    # Drop the table
    op.drop_table('subscription_access')

    # Drop the ENUM type
    subscription_access_status_enum = sa.Enum(
        'ACTIVE', 'DISABLED', name='subscription_access_status_enum'
    )
    subscription_access_status_enum.drop(op.get_bind(), checkfirst=True)
