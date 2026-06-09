"""add status column to gitlab-webhook table

Revision ID: 033
Revises: 032
Create Date: 2025-04-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '033'
down_revision: Union[str, None] = '032'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'gitlab_webhook', sa.Column('webhook_uuid', sa.String(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('gitlab_webhook', 'webhook_uuid')
