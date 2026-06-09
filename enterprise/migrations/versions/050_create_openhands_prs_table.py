"""Create openhands_prs table

Revision ID: 050
Revises: 049
Create Date: 2025-06-18

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '050'
down_revision: Union[str, None] = '049'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types
    op.create_table(
        'openhands_prs',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('repo_id', sa.String(), nullable=False),
        sa.Column('repo_name', sa.String(), nullable=False),
        sa.Column('pr_number', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('installation_id', sa.String(), nullable=True),
        sa.Column('private', sa.Boolean(), nullable=True),
        sa.Column(
            'status', sa.Enum('MERGED', 'CLOSED', name='prstatus'), nullable=False
        ),
        sa.Column(
            'processed', sa.Boolean(), nullable=False, server_default=sa.text('FALSE')
        ),
        sa.Column(
            'process_attempts',
            sa.Integer(),
            nullable=False,
            server_default=sa.text('0'),
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(),
            nullable=False,
        ),
        sa.Column(
            'closed_at',
            sa.DateTime(),
            nullable=False,
        ),
        # PR metrics columns (optional fields)
        sa.Column('num_reviewers', sa.Integer(), nullable=True),
        sa.Column('num_commits', sa.Integer(), nullable=True),
        sa.Column('num_review_comments', sa.Integer(), nullable=True),
        sa.Column('num_general_comments', sa.Integer(), nullable=True),
        sa.Column('num_changed_files', sa.Integer(), nullable=True),
        sa.Column('num_additions', sa.Integer(), nullable=True),
        sa.Column('num_deletions', sa.Integer(), nullable=True),
        sa.Column('merged', sa.Boolean(), nullable=True),
        sa.Column('openhands_helped_author', sa.Boolean(), nullable=True),
        sa.Column('num_openhands_commits', sa.Integer(), nullable=True),
        sa.Column('num_openhands_review_comments', sa.Integer(), nullable=True),
        sa.Column('num_openhands_general_comments', sa.Integer(), nullable=True),
    )

    # Create indexes for efficient querying
    op.create_index(
        'ix_openhands_prs_repo_id', 'openhands_prs', ['repo_id'], unique=False
    )
    op.create_index(
        'ix_openhands_prs_pr_number', 'openhands_prs', ['pr_number'], unique=False
    )
    op.create_index(
        'ix_openhands_prs_status', 'openhands_prs', ['status'], unique=False
    )

    # Create unique constraint on repo_id + pr_number combination
    op.create_index(
        'ix_openhands_prs_repo_pr_unique',
        'openhands_prs',
        ['repo_id', 'pr_number'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('ix_openhands_prs_repo_id', table_name='openhands_prs')
    op.drop_index('ix_openhands_prs_pr_number', table_name='openhands_prs')
    op.drop_index('ix_openhands_prs_status', table_name='openhands_prs')
    op.drop_index('ix_openhands_prs_repo_pr_unique', table_name='openhands_prs')
    op.drop_table('openhands_prs')

    # Drop enum types
    op.execute('DROP TYPE IF EXISTS prstatus')
    op.execute('DROP TYPE IF EXISTS providertype')
