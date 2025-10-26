"""Create user version upgrade tasks

Revision ID: 060
Revises: 059
Create Date: 2025-07-21

This migration creates maintenance tasks for upgrading user versions
to replace the removed admin maintenance endpoint.
"""

import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.
revision = '060'
down_revision = '059'
branch_labels = None
depends_on = None

# TODO: decide whether to modify this for orgs or users


def upgrade():
    """
    Create maintenance tasks for all users whose user_version is less than
    the current version.

    This replaces the functionality of the removed admin maintenance endpoint.
    """

    # Hardcoded value to prevent migration failures when constant is removed from codebase
    # This migration has already run in production, so we use the value that was current at the time
    CURRENT_USER_SETTINGS_VERSION = 4

    # Create a connection and bind it to a session
    connection = op.get_bind()
    session = Session(bind=connection)

    try:
        # Find all users that need upgrading
        users_needing_upgrade = session.execute(
            sa.text(
                'SELECT keycloak_user_id FROM user_settings WHERE user_version < :current_version'
            ),
            {'current_version': CURRENT_USER_SETTINGS_VERSION},
        ).fetchall()

        if not users_needing_upgrade:
            # No users need upgrading
            return

        # Get user IDs
        user_ids = [user[0] for user in users_needing_upgrade]

        # Create tasks in batches of 100 users each (as per processor limit)
        # Space the start time for batches a minute apart to distribute the load
        batch_size = 100

        for i in range(0, len(user_ids), batch_size):
            batch_user_ids = user_ids[i : i + batch_size]

            # Calculate start time for this batch (space batches 1 minute apart)

            # Create processor JSON
            processor_type = 'server.maintenance_task_processor.user_version_upgrade_processor.UserVersionUpgradeProcessor'
            processor_json = json.dumps({'user_ids': batch_user_ids})

            # Insert maintenance task directly
            session.execute(
                sa.text(
                    """
                    INSERT INTO maintenance_tasks
                    (status, processor_type, processor_json, delay, created_at, updated_at)
                    VALUES
                    ('PENDING', :processor_type, :processor_json, :delay, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """
                ),
                {
                    'processor_type': processor_type,
                    'processor_json': processor_json,
                    'delay': 10,
                },
            )

        # Commit all tasks
        session.commit()

    finally:
        session.close()


def downgrade():
    """
    No downgrade operation needed as we're just creating tasks.
    The tasks themselves will be processed and completed.

    If needed, we could delete tasks with this processor type, but that's not necessary
    since they're meant to be processed and completed.
    """
    pass
