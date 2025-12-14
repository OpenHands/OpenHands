"""Add public sharing fields to conversation metadata

Revision ID: 082
Revises: 081
Create Date: 2025-01-27 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '082'
down_revision = '081'
branch_labels = None
depends_on = None


def upgrade():
    """Add public sharing fields to conversation_metadata table."""
    # Add is_public column with default False and index
    op.add_column('conversation_metadata', 
                  sa.Column('is_public', sa.Boolean(), nullable=False, default=False))
    
    # Add public_share_token column with index for efficient lookups
    op.add_column('conversation_metadata', 
                  sa.Column('public_share_token', sa.String(), nullable=True))
    
    # Add shared_at timestamp column
    op.add_column('conversation_metadata', 
                  sa.Column('shared_at', sa.DateTime(timezone=True), nullable=True))
    
    # Create indexes for efficient queries
    op.create_index('ix_conversation_metadata_is_public', 'conversation_metadata', ['is_public'])
    op.create_index('ix_conversation_metadata_public_share_token', 'conversation_metadata', ['public_share_token'])


def downgrade():
    """Remove public sharing fields from conversation_metadata table."""
    # Drop indexes first
    op.drop_index('ix_conversation_metadata_public_share_token', 'conversation_metadata')
    op.drop_index('ix_conversation_metadata_is_public', 'conversation_metadata')
    
    # Drop columns
    op.drop_column('conversation_metadata', 'shared_at')
    op.drop_column('conversation_metadata', 'public_share_token')
    op.drop_column('conversation_metadata', 'is_public')