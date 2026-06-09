"""create org_invitation table

Revision ID: 094
Revises: 093
Create Date: 2026-02-18 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '094'
down_revision: Union[str, None] = '093'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create org_invitation table
    op.create_table(
        'org_invitation',
        sa.Column('id', sa.Integer, sa.Identity(), primary_key=True),
        sa.Column('token', sa.String(64), nullable=False),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('role_id', sa.Integer, nullable=False),
        sa.Column('inviter_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            'status',
            sa.String(20),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            'created_at',
            sa.DateTime,
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
        ),
        sa.Column('expires_at', sa.DateTime, nullable=False),
        sa.Column('accepted_at', sa.DateTime, nullable=True),
        sa.Column('accepted_by_user_id', postgresql.UUID(as_uuid=True), nullable=True),
        # Foreign key constraints
        sa.ForeignKeyConstraint(
            ['org_id'],
            ['org.id'],
            name='org_invitation_org_fkey',
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['role_id'],
            ['role.id'],
            name='org_invitation_role_fkey',
        ),
        sa.ForeignKeyConstraint(
            ['inviter_id'],
            ['user.id'],
            name='org_invitation_inviter_fkey',
        ),
        sa.ForeignKeyConstraint(
            ['accepted_by_user_id'],
            ['user.id'],
            name='org_invitation_accepter_fkey',
        ),
    )

    # Create indexes
    op.create_index(
        'ix_org_invitation_token',
        'org_invitation',
        ['token'],
        unique=True,
    )
    op.create_index(
        'ix_org_invitation_org_id',
        'org_invitation',
        ['org_id'],
    )
    op.create_index(
        'ix_org_invitation_email',
        'org_invitation',
        ['email'],
    )
    op.create_index(
        'ix_org_invitation_status',
        'org_invitation',
        ['status'],
    )
    # Composite index for checking pending invitations
    op.create_index(
        'ix_org_invitation_org_email_status',
        'org_invitation',
        ['org_id', 'email', 'status'],
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_org_invitation_org_email_status', table_name='org_invitation')
    op.drop_index('ix_org_invitation_status', table_name='org_invitation')
    op.drop_index('ix_org_invitation_email', table_name='org_invitation')
    op.drop_index('ix_org_invitation_org_id', table_name='org_invitation')
    op.drop_index('ix_org_invitation_token', table_name='org_invitation')

    # Drop table
    op.drop_table('org_invitation')
