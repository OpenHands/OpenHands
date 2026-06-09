"""add type column to billing_sessions

Revision ID: 073
Revises: 072
Create Date: 2025-08-26

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '073'
down_revision: Union[str, None] = '072'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the ENUM type explicitly, then add the column using it
    billing_session_type_enum = sa.Enum(
        'DIRECT_PAYMENT', 'MONTHLY_SUBSCRIPTION', name='billing_session_type_enum'
    )
    billing_session_type_enum.create(op.get_bind(), checkfirst=True)

    op.add_column(
        'billing_sessions',
        sa.Column(
            'billing_session_type',
            billing_session_type_enum,
            nullable=False,
            server_default='DIRECT_PAYMENT',
        ),
    )


def downgrade() -> None:
    # Drop the column then drop the ENUM type
    op.drop_column('billing_sessions', 'billing_session_type')
    billing_session_type_enum = sa.Enum(
        'DIRECT_PAYMENT', 'MONTHLY_SUBSCRIPTION', name='billing_session_type_enum'
    )
    billing_session_type_enum.drop(op.get_bind(), checkfirst=True)
