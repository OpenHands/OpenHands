"""Rename 'user' role to 'member' in role table.

Revision ID: 092
Revises: 091
Create Date: 2025-02-12 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '092'
down_revision: Union[str, None] = '091'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename 'user' role to 'member' for clarity
    # This avoids confusion between the 'user' role and the 'user' entity/account
    op.execute(sa.text("UPDATE role SET name = 'member' WHERE name = 'user'"))


def downgrade() -> None:
    # Revert 'member' role back to 'user'
    op.execute(sa.text("UPDATE role SET name = 'user' WHERE name = 'member'"))
