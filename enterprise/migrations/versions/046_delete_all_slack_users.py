"""delete all slack users

Revision ID: 046
Revises: 045
Create Date: 2025-06-11 18:11:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '046'
down_revision: Union[str, None] = '045'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Delete all rows from the slack_users table
    op.execute('DELETE FROM slack_users')


def downgrade() -> None:
    # Cannot restore deleted data
    pass
