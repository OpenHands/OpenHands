"""Add personal_skills_repo columns to user_settings.

Revision ID: 113
Revises: 112
Create Date: 2026-05-02 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '113'
down_revision: Union[str, None] = '112'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings',
        sa.Column('personal_skills_repo_url', sa.String(), nullable=True),
    )
    op.add_column(
        'user_settings',
        sa.Column('personal_skills_repo_commit', sa.String(), nullable=True),
    )
    op.add_column(
        'user_settings',
        sa.Column('personal_skills_repo_updated_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'personal_skills_repo_updated_at')
    op.drop_column('user_settings', 'personal_skills_repo_commit')
    op.drop_column('user_settings', 'personal_skills_repo_url')
