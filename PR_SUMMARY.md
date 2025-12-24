# Pull Request Summary

## Description
This PR fixes critical issues that prevent OpenHands from running properly in containerized environments (Dev Containers, docker-compose deployments) and resolves SQLite database migration errors.

## Issues Fixed

### 1. Sandbox Connection Failures in Containerized Environments
**Problem:**
- Backend running inside Docker containers (Dev Containers, docker-compose) cannot communicate with sandbox containers
- Error: `Sandbox failed to start: oh-agent-server-xxxxx`
- Error: `http://host.docker.internal:XXXXX : Server disconnected without sending a response`
- Error: `All connection attempts failed`

**Root Cause (Network Namespace Isolation):**
When the backend runs in a container, it faces a fundamental networking issue:
1. Backend container dynamically creates sandbox containers and binds their ports to `localhost` in the container's network namespace
2. OpenHands detects it's running in Docker and converts URLs from `localhost` to `host.docker.internal` for cross-container communication
3. However, `host.docker.internal` resolves to the **host machine**, not the container's `localhost`
4. The sandbox ports are bound to the **container's localhost**, creating a namespace mismatch
5. Result: Backend cannot reach sandbox even though both are running

This is **not** a DNS resolution issue. The project's docker-compose.yml already includes `extra_hosts: - "host.docker.internal:host-gateway"` for DNS, but this doesn't solve the network namespace isolation problem.

**Solution:**
Using `use_host_network = true` makes sandbox containers share the host's network namespace, eliminating the isolation issue and allowing reliable communication.

### 2. SQLite Database Migration Errors
**Problem:**
- Migration failures with `sqlite3.OperationalError: near "DROP": syntax error`
- Partial migrations leaving database in inconsistent state
- Error: `duplicate column name: status`

**Root Cause:**
SQLite versions < 3.35.0 do not support `DROP COLUMN` operation directly. The migration was using standard `op.drop_column()` which generates incompatible SQL.

**Solution:**
Updated migration file to use SQLite-compatible `batch_alter_table()` pattern for column modifications.

## Changes Made

### 1. Fixed SQLite Migration (`openhands/app_server/app_lifespan/alembic/versions/002.py`)
- ✅ Replaced `op.drop_column()` with `batch_alter_table()` context manager
- ✅ Now compatible with older SQLite versions
- ✅ Ensures transactional safety for schema changes
- ✅ Applied to both `upgrade()` and `downgrade()` functions

**Before:**
```python
op.drop_index('ix_event_callback_result_event_id')
op.drop_column('event_callback_result', 'event_id')
op.add_column('event_callback_result', sa.Column('event_id', sa.String, nullable=True))
```

**After:**
```python
with op.batch_alter_table('event_callback_result', schema=None) as batch_op:
    batch_op.drop_index('ix_event_callback_result_event_id')
    batch_op.drop_column('event_id')
    batch_op.add_column(sa.Column('event_id', sa.String, nullable=True))
```

### 2. Enhanced Configuration Documentation (`config.template.toml`)
- ✅ Added detailed comments explaining `use_host_network` option
- ✅ Clear examples for different deployment scenarios
- ✅ Explains when to enable/disable the option

### 3. Created Comprehensive Troubleshooting Guide (`TROUBLESHOOTING.md`)
- ✅ Step-by-step solutions for sandbox connection errors
- ✅ Database migration error recovery procedures
- ✅ Environment-specific configuration guidance
- ✅ Verification steps and diagnostics
- ✅ Quick reference table for different environments

### 4. Updated Development Documentation (`Development.md`)
- ✅ Added warning for Dev Container users
- ✅ Links to troubleshooting guide for quick reference

### 5. Created Production Config Template (`config.production.toml`)
- ✅ Example configuration for containerized production deployments
- ✅ Includes explanatory comments

## Testing

### Environments Tested
- ✅ Dev Container (VS Code Remote Containers)
- ✅ Docker Compose production deployment
- ✅ Local development (native Python)

### Verification Steps
1. Database migration runs successfully without errors
2. Sandbox containers start and connect properly
3. Backend can communicate with sandbox via health check endpoint
4. Conversations can be started without "Sandbox failed to start" errors

## Impact

### Who Benefits
- **Dev Container Users:** Can now run OpenHands without sandbox connection errors
- **Docker Compose Deployments:** Proper configuration guidance for containerized production setups
- **New Contributors:** Clear troubleshooting documentation reduces friction
- **All Users:** SQLite migrations work reliably across different SQLite versions

### Backward Compatibility
- ✅ No breaking changes
- ✅ Default configuration unchanged (`use_host_network = false`)
- ✅ Database migration is forward-compatible
- ✅ Existing deployments continue to work

## Documentation Changes
- ✅ `TROUBLESHOOTING.md` (new file)
- ✅ `config.template.toml` (enhanced comments)
- ✅ `Development.md` (added troubleshooting reference)
- ✅ `config.production.toml` (new example file)

## Related Issues
- Fixes issues with sandbox initialization failures in containerized environments
- Resolves SQLite migration errors reported by users
- Improves developer experience for contributors using Dev Containers

## Checklist
- [x] Code changes tested in multiple environments
- [x] Documentation updated
- [x] No breaking changes introduced
- [x] Migration tested with both upgrade and downgrade
- [x] Configuration examples provided
- [x] Troubleshooting guide created

---

## For Reviewers

**Key files to review:**
1. `openhands/app_server/app_lifespan/alembic/versions/002.py` - Migration fix
2. `TROUBLESHOOTING.md` - New troubleshooting guide
3. `config.template.toml` - Enhanced documentation

**Testing suggestions:**
1. Run in Dev Container and verify sandbox starts
2. Delete database and verify migrations run cleanly
3. Review documentation for clarity and completeness
