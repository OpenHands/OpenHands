"""Create org_git_claim table for tracking Git organization claims.

Revision ID: 105
Revises: 104
Create Date: 2026-04-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '105'
down_revision: Union[str, None] = '104'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'org_git_claim',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('org_id', sa.UUID(), nullable=False),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('git_organization', sa.String(), nullable=False),
        sa.Column('claimed_by', sa.UUID(), nullable=False),
        sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['org_id'], ['org.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['claimed_by'], ['user.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('provider', 'git_organization', name='uq_provider_git_org'),
    )


def downgrade() -> None:
    op.drop_table('org_git_claim')
