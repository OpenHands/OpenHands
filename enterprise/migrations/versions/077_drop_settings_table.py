"""drop settings table

Revision ID: 077
Revises: 076
Create Date: 2025-10-21 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '077'
down_revision: Union[str, None] = '076'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop the deprecated settings table."""
    op.execute('DROP TABLE IF EXISTS settings')


def downgrade() -> None:
    """No-op downgrade since the settings table is deprecated."""
    pass
