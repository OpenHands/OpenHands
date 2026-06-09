"""Create feedback table

Revision ID: 001
Revises:
Create Date: 2024-03-19 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'feedback',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column(
            'polarity',
            sa.Enum('positive', 'negative', name='polarity_enum'),
            nullable=False,
        ),
        sa.Column(
            'permissions',
            sa.Enum('public', 'private', name='permissions_enum'),
            nullable=False,
        ),
        sa.Column('trajectory', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('feedback')
    op.execute('DROP TYPE polarity_enum')
    op.execute('DROP TYPE permissions_enum')
