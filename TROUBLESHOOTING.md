# Troubleshooting Guide

This guide helps resolve common issues when running OpenHands.

## Table of Contents
- [Sandbox Connection Errors](#sandbox-connection-errors)
- [Database Migration Errors](#database-migration-errors)

---

## Sandbox Connection Errors

### Symptom: "Sandbox failed to start" or "Server disconnected without sending a response"

**Error Examples:**
```
Sandbox server not running: http://host.docker.internal:37879 : Server disconnected without sending a response
SandboxError: 500: Sandbox failed to start: oh-agent-server-xxxxx
```

**Root Cause:**
When the OpenHands backend runs inside a Docker container (Dev Container, docker-compose, etc.), it needs to communicate with sandbox containers that it dynamically creates. The system converts `localhost` URLs to `host.docker.internal` to enable this communication. However, even with `extra_hosts` configuration in docker-compose.yml, the following issue occurs:

**Technical Details:**
1. Backend container binds sandbox ports to `localhost` on the **container's network namespace**
2. System converts `localhost` → `host.docker.internal` for Docker environments
3. `host.docker.internal` points to the **host machine**, not the container's localhost
4. Result: Backend tries to reach sandbox via host machine, but ports are bound to container's localhost
5. Connection fails with "Server disconnected" or "All connection attempts failed"

This is a **network namespace isolation issue** specific to containerized backends, not a DNS resolution problem.

### Solution

#### Enable Host Network Mode (Required for Containerized Backends)

Add the following to your `config.toml`:

```toml
[sandbox]
use_host_network = true
```

**Automatic Configuration:**
If you run `make setup-config` in a containerized environment (Dev Container, docker-compose), this setting is automatically added for you. You'll see a confirmation message during setup.

**Manual Configuration:**
If you already have a `config.toml` without this setting, simply add the `[sandbox]` section with `use_host_network = true` as shown above.

**How it works:**
- Sandbox containers share the host's network namespace
- Eliminates network isolation between backend and sandbox
- Both can communicate via `localhost` or `host.docker.internal` reliably

**When to use:**
- ✅ Running in Dev Containers / GitHub Codespaces
- ✅ Backend deployed via docker-compose (containerized)
- ✅ Any scenario where the OpenHands backend runs inside a Docker container
- ✅ Single-tenant deployments or controlled environments

**When NOT to use:**
- ❌ Running backend natively (Python directly on host OS) - default config works
- ❌ Multi-tenant environments requiring strict network isolation

**Note:** The project's docker-compose.yml already includes `extra_hosts: - "host.docker.internal:host-gateway"`, but this alone is insufficient for containerized backends due to network namespace isolation.

#### Verification

After applying the fix:

1. **Stop existing containers:**
   ```bash
   docker rm -f $(docker ps -aq --filter "name=oh-agent-server") 2>/dev/null || true
   ```

2. **Restart OpenHands:**
   ```bash
   # For development
   make run

   # For docker-compose
   docker-compose down && docker-compose up
   ```

3. **Check logs:**
   ```bash
   # Development
   tail -f logs/*.log

   # Docker Compose
   docker-compose logs -f
   ```

4. **Verify sandbox is accessible:**
   ```bash
   # Find the sandbox port (look for logs mentioning port mapping)
   docker ps | grep oh-agent-server

   # Test the health endpoint
   curl http://localhost:<SANDBOX_PORT>/alive
   # Should return: {"status":"ok"}
   ```

---

## Database Migration Errors

### Symptom: SQLite errors during application startup

**Error Examples:**
```
sqlite3.OperationalError: near "DROP": syntax error
[SQL: ALTER TABLE event_callback_result DROP COLUMN event_id]

sqlite3.OperationalError: duplicate column name: status
[SQL: ALTER TABLE event_callback ADD COLUMN status VARCHAR(9)]
```

**Root Cause:**
The database is in a partially migrated state, typically from:
- Interrupted migration process
- SQLite version incompatibility
- Corrupted database file

### Solution

#### Step 1: Remove Corrupted Database

```bash
# Find the database location (usually in ~/.openhands/)
rm -f ~/.openhands/openhands.db

# Or if persistence_dir is custom, check your config
# rm -f /path/to/your/persistence_dir/openhands.db
```

#### Step 2: Restart Application

The database will be recreated automatically with the correct schema on next startup:

```bash
make run
```

#### Verification

Check the logs for successful migration:

```
INFO:alembic.runtime.migration:Running upgrade  -> 001, Sync DB with Models
INFO:alembic.runtime.migration:Running upgrade 001 -> 002, Sync DB with Models
INFO:alembic.runtime.migration:Running upgrade 002 -> 003, add parent_conversation_id
INFO:     Application startup complete.
```

### Additional Database Notes

**For Development:**
- Database resets are safe and expected
- No critical data is stored in the local database

**For Production:**
- Consider backing up the database before major updates:
  ```bash
  cp ~/.openhands/openhands.db ~/.openhands/openhands.db.backup
  ```
- Use PostgreSQL for production deployments instead of SQLite

---

## Environment-Specific Issues

### Dev Containers (VS Code Remote Containers)

**Issue:** Network isolation prevents container communication

**Solution:**
```toml
[sandbox]
use_host_network = true
```

### GitHub Codespaces

**Issue:** Same as Dev Containers

**Solution:**
```toml
[sandbox]
use_host_network = true
```

### Docker Compose Production Deployment

**Issue:** Backend container cannot reach sandbox containers due to network namespace isolation

**Solution:**
```toml
[sandbox]
use_host_network = true
```

**Why the default docker-compose.yml config is not enough:**
The project includes `extra_hosts: - "host.docker.internal:host-gateway"` by default, but this only helps with DNS resolution. The real issue is that sandbox ports are bound to the container's localhost, while `host.docker.internal` points to the host machine, creating a network namespace mismatch.

### Native Python Development (No Containers)

**Configuration:**
```toml
# No special sandbox configuration needed
# Default settings work fine
```

---

## Getting Help

If these solutions don't resolve your issue:

1. **Check existing issues:** [OpenHands Issues](https://github.com/OpenHands/OpenHands/issues)
2. **Gather information:**
   - OpenHands version: `cat openhands/version.py`
   - Python version: `python --version`
   - Docker version: `docker --version`
   - OS and environment (Dev Container, native, docker-compose, etc.)
   - Full error logs

3. **Create a new issue** with the gathered information

---

## Quick Reference

| Environment | `use_host_network` | Notes |
|-------------|-------------------|-------|
| Dev Container | `true` | Required due to container nesting |
| GitHub Codespaces | `true` | Same as Dev Container |
| Docker Compose (containerized) | `true` | Backend in container needs access to sandbox |
| Native Python | `false` | Default works fine |
| Kubernetes | `false` | Use cluster DNS instead |

---

*Last updated: December 2025*
