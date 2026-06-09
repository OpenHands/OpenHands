"""add cancellation fields to subscription_access

Revision ID: 075
Revises: 074
Create Date: 2025-01-11

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '075'
down_revision: Union[str, None] = '074'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add cancelled_at field to track cancellation timestamp
    op.add_column(
        'subscription_access',
        sa.Column('cancelled_at', sa.DateTime(timezone=True), nullable=True),
    )

    # Add stripe_subscription_id field to enable cancellation via Stripe API
    op.add_column(
        'subscription_access',
        sa.Column('stripe_subscription_id', sa.String(), nullable=True),
    )

    # Create index on stripe_subscription_id for efficient lookups
    op.create_index(
        'ix_subscription_access_stripe_subscription_id',
        'subscription_access',
        ['stripe_subscription_id'],
    )


def downgrade() -> None:
    # Drop index
    op.drop_index(
        'ix_subscription_access_stripe_subscription_id', 'subscription_access'
    )

    # Drop columns
    op.drop_column('subscription_access', 'stripe_subscription_id')
    op.drop_column('subscription_access', 'cancelled_at')
