# Maintenance Task System

This package contains the maintenance task system for running background maintenance operations in the OpenHands deployment wrapper.

## Overview

The maintenance task system provides a framework for running background tasks that perform maintenance operations such as upgrading user settings, cleaning up data, or other periodic maintenance work. Tasks are designed to be short-running (typically under a minute) and handle background state upgrades. The runner is triggered as part of every deploy, though does not block it.

## Architecture

The system consists of several key components:

### 1. Database Model (`MaintenanceTask`)

Located in `storage/maintenance_task.py`, this model stores maintenance tasks with the following fields:

- `id`: Primary key
- `status`: Task status (INACTIVE, PENDING, WORKING, COMPLETED, ERROR)
- `processor_type`: Fully qualified class name of the processor
- `processor_json`: JSON serialized processor configuration
- `delay`: Delay before starting task
- `info`: JSON field containing structured information about the task outcome
- `created_at`: When the task was created
- `updated_at`: When the task was last updated

### 2. Processor Base Class (`MaintenanceTaskProcessor`)

Abstract base class for all maintenance task processors. Processors must implement the `__call__` method to perform the actual work.

```python
from storage.maintenance_task import MaintenanceTaskProcessor, MaintenanceTask

class MyProcessor(MaintenanceTaskProcessor):
    # Define your processor fields here
    some_config: str

    async def __call__(self, task: MaintenanceTask) -> dict:
        # Implement your maintenance logic here
        return {"status": "completed", "processed_items": 42}
```

## Available Processors

### UserVersionUpgradeProcessor

Located in `user_version_upgrade_processor.py`, this processor:

- Handles up to 100 user IDs per task
- Upgrades users with `user_version < ORG_SETTINGS_VERSION`
- Uses `SaasSettingsStore.create_default_settings()` for upgrades

**Usage:**

```python
from server.maintenance_task_processor.user_version_upgrade_processor import UserVersionUpgradeProcessor

processor = UserVersionUpgradeProcessor(user_ids=["user1", "user2", "user3"])
```

## Creating New Processors

To create a new maintenance task processor:

1. **Create a new processor class** inheriting from `MaintenanceTaskProcessor`:

```python
from storage.maintenance_task import MaintenanceTaskProcessor, MaintenanceTask
from typing import List

class MyMaintenanceProcessor(MaintenanceTaskProcessor):
    """Description of what this processor does."""

    # Define configuration fields
    target_ids: List[str]
    batch_size: int = 50

    async def __call__(self, task: MaintenanceTask) -> dict:
        """
        Implement your maintenance logic here.

        Args:
            task: The maintenance task being processed

        Returns:
            dict: Information about the task execution
        """
        try:
            # Your maintenance logic here
            processed_count = 0

            for target_id in self.target_ids:
                # Process each target
                processed_count += 1

            return {
                "status": "completed",
                "processed_count": processed_count,
                "message": f"Successfully processed {processed_count} items"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "processed_count": processed_count
            }
```

2. **Add the processor to the package** by importing it in `__init__.py` if needed.

3. **Create tasks using the utility functions** in `server/utils/maintenance_task_utils.py`:

```python
from server.utils.maintenance_task_utils import create_maintenance_task
from server.maintenance_task_processor.my_processor import MyMaintenanceProcessor

# Create a task
processor = MyMaintenanceProcessor(target_ids=["id1", "id2"], batch_size=25)
task = create_maintenance_task(processor, start_at=datetime.utcnow())
```

## Task Management

### Creating Tasks Programmatically

```python
from datetime import datetime, timedelta
from server.utils.maintenance_task_utils import create_maintenance_task
from server.maintenance_task_processor.user_version_upgrade_processor import UserVersionUpgradeProcessor

# Create a user upgrade task
processor = UserVersionUpgradeProcessor(user_ids=["user1", "user2"])
task = create_maintenance_task(
    processor=processor,
    start_at=datetime.utcnow() + timedelta(minutes=5)  # Start in 5 minutes
)
```

## Task Lifecycle

1. **INACTIVE**: Task is created but not yet scheduled
2. **PENDING**: Task is scheduled and waiting to be picked up by the runner
3. **WORKING**: Task is currently being processed
4. **COMPLETED**: Task finished successfully
5. **ERROR**: Task encountered an error during processing

## Best Practices

### Processor Design

- Keep tasks short-running (under 1 minute)
- Handle errors gracefully and return meaningful error information
- Use batch processing for large datasets
- Include progress information in the return dict

### Error Handling

- Always wrap your processor logic in try-catch blocks
- Return structured error information
- Log important events for debugging

### Performance

- Limit batch sizes to avoid long-running tasks
- Use database sessions efficiently
- Consider memory usage for large datasets

### Testing

- Create unit tests for your processors
- Test error conditions
- Verify the processor serialization/deserialization works correctly

## Database Patterns

The maintenance task system follows the repository's established patterns:

- Uses `session_maker()` for database operations
- Wraps sync database operations in `call_sync_from_async` for async routes
- Follows proper SQLAlchemy query patterns

## Integration with Existing Systems

### User Management

- Integrates with the existing `UserSettings` model
- Uses the current user versioning system (`ORG_SETTINGS_VERSION`)
- Maintains compatibility with existing user management workflows

### Authentication

- Admin endpoints use the existing SaaS authentication system
- Requires users to have `admin = True` in their UserSettings

### Monitoring

- Tasks are logged with structured information
- Status updates are tracked in the database
- Error information is preserved for debugging

## Troubleshooting

### Common Issues

1. **Tasks stuck in WORKING state**: Usually indicates the runner crashed while processing. These can be manually reset to PENDING.

2. **Serialization errors**: Ensure all processor fields are JSON serializable.

3. **Database connection issues**: Check that the processor properly handles database sessions.

### Debugging

- Check the server logs for task execution details
- Use the admin API to inspect task status and info
- Verify processor configuration is correct

## Future Enhancements

Potential improvements that could be added:

- Task dependencies and scheduling
- Retry mechanisms for failed tasks
- Real-time progress updates
- Task cancellation
- Cron-like scheduling expressions
- Audit logging for admin actions
- Role-based permissions beyond simple admin flag
