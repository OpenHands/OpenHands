"""create user-repos table

Revision ID: 027
Revises: 026
Create Date: 2025-04-14

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '028'
down_revision: Union[str, None] = '027'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'repos',
        sa.Column('id', sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column('repo_name', sa.String(), nullable=False),
        sa.Column('repo_id', sa.String(), nullable=False),
        sa.Column('is_public', sa.Boolean(), nullable=False),
        sa.Column('has_microagent', sa.Boolean(), nullable=True),
        sa.Column('has_setup_script', sa.Boolean(), nullable=True),
    )

    op.create_index(
        'idx_repos_repo_id',
        'repos',
        ['repo_id'],
    )

    op.create_table(
        'user-repos',
        sa.Column('id', sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('repo_id', sa.String(), nullable=False),
        sa.Column('admin', sa.Boolean(), nullable=True),
    )

    op.create_index(
        'idx_user_repos_repo_id',
        'user-repos',
        ['repo_id'],
    )
    op.create_index(
        'idx_user_repos_user_id',
        'user-repos',
        ['user_id'],
    )


def downgrade() -> None:
    op.drop_index('idx_repos_repo_id', 'repos')
    op.drop_index('idx_user_repos_repo_id', 'user-repos')
    op.drop_index('idx_user_repos_user_id', 'user-repos')
    op.drop_table('repos')
    op.drop_table('user-repos')
