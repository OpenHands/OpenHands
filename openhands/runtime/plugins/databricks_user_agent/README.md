# Databricks User-Agent Plugin

## Overview

This runtime plugin applies a consistent `User-Agent` to Databricks API traffic
made from inside the OpenHands runtime, so calls from OpenHands are identifiable
in Databricks audit logs.

The identity is **fixed**: `OpenHandsOSS/<version>`, where `<version>` is sourced
from the installed `openhands-sdk` package. This keeps a single, consistent
identity across the AI Gateway HTTP path, OAuth, and the Databricks SDK — it is
not user-configurable. The LLM connector (in `openhands-sdk`) already sets this
same User-Agent on its own requests; this plugin extends the same attribution to
non-LLM Databricks API calls made from the runtime (e.g. the `databricks-sdk`,
direct REST calls, and other HTTP clients).

## What it configures

When the plugin initializes it:

1. **Sets environment variables** the Databricks SDK recognizes:
   - `DATABRICKS_SDK_UPSTREAM` = `OpenHandsOSS`
   - `DATABRICKS_SDK_UPSTREAM_VERSION` = the `openhands-sdk` version
   - `DATABRICKS_USER_AGENT` = `OpenHandsOSS`
   - `OH_DATABRICKS_INTEGRATION` = `true` (marker that the integration is active)
2. **Patches Python HTTP libraries** by writing a startup script
   (`~/.databricks_user_agent_init.py`) that injects the User-Agent for
   `requests`, `urllib3`, `httpx`, and `aiohttp`, and wires it via the shell
   environment (`PYTHONSTARTUP`).
3. **Configures Java** (when `configure_java=True`) via `JAVA_TOOL_OPTIONS`.
4. **Writes helper/test scripts** into the user's home directory, including
   `~/.databricks_user_agent_test.py` for verification.

Initialization failures are logged but **non-fatal** — the runtime continues
even if the plugin can't set things up.

## Configuration

The plugin exposes a small `DatabricksUserAgentRequirement` dataclass. The
User-Agent string itself is **not** configurable; only these toggles are:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"databricks_user_agent"` | Plugin identifier |
| `patch_http_libraries` | bool | `True` | Patch Python HTTP libraries (`requests`/`urllib3`/`httpx`/`aiohttp`) |
| `configure_java` | bool | `True` | Configure Java User-Agent via `JAVA_TOOL_OPTIONS` |
| `enable_debug_logging` | bool | `False` | Verbose plugin logging |

`user_agent` and `version` are exposed as **read-only properties** derived from
`openhands-sdk`; they are not constructor arguments.

## Usage

```python
from openhands.runtime.plugins.databricks_user_agent import (
    DatabricksUserAgentRequirement,
    DatabricksUserAgentPlugin,
    create_databricks_plugin,
)

# Default requirement (recommended)
requirement = DatabricksUserAgentRequirement()

# Or with debug logging via the convenience constructor:
plugin = create_databricks_plugin(enable_debug=True)
```

The plugin implements the runtime plugin interface (a `Requirement` plus a
`Plugin` whose async `initialize(username)` performs the setup above). It is
activated by including the requirement in the runtime's plugin list; it is not
auto-registered today.

## Verification

After the runtime starts, run the generated test script inside the runtime:

```bash
python3 ~/.databricks_user_agent_test.py
```

It prints the resolved environment variables and confirms which HTTP libraries
were patched. You can also confirm attribution from the Databricks side:

1. Databricks workspace → Admin Console → Audit Logs
2. Inspect the `userAgent` field on API requests originating from OpenHands;
   it should read `OpenHandsOSS/<version>`.

## Security

- The User-Agent is visible in Databricks audit logs and to any intermediary
  proxies — it carries no secrets, only the `OpenHandsOSS/<version>` identity.
- The plugin only modifies HTTP headers / environment, not request payloads.

## Contributing

1. Code: `openhands/runtime/plugins/databricks_user_agent/__init__.py`
2. Tests: `tests/unit/runtime/plugins/test_databricks_user_agent.py`
3. Keep this README in sync with the dataclass fields and the fixed
   `OpenHandsOSS/<version>` identity.
