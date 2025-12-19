### Fleet Runtime

The Fleet Runtime allows OpenHands to execute actions within remote environments provisioned by [Fleet](https://github.com/fleet-ai/OpenEnv). It uses a split-plane architecture:
- **Orchestration (HTTP)**: Resets and manages environment state.
- **Actions (MCP)**: Discovers and calls tools (e.g., `bash`, `computer`, `str_replace_editor`) exposed by the Fleet environment via the Model Context Protocol.

#### Configuration

To use the Fleet Runtime, update your `config.toml`:

```toml
[core]
runtime = "fleet"

[sandbox]
fleet_api_key = "fl_..."
fleet_env_key = "ubuntu"  # or any other Fleet environment key
```

#### Architecture

Unlike `DockerRuntime` which spins up a local container with an `ActionExecutionServer`, `FleetRuntime` acts as a bridge:
1.  **Connects** to Fleet using `FleetEnvClient`.
2.  **Discovers** tools (capabilities) using `FleetMCPTools`.
3.  **Maps** OpenHands actions (`CmdRunAction`, `FileReadAction`) to the corresponding Fleet MCP tools.
4.  **Forwards** generic `MCPAction` calls directly to the Fleet environment.

#### Observability

- **Local Logging**: `FleetRuntime` logs tool discovery and execution metrics (latency, success/failure) to the console.
- **Remote Tracing**: You can export action/observation traces to an external API by configuring `trace_export_url` in `config.toml`.

#### Requirements

You must install the Fleet SDK integration:
```bash
pip install "openenv-core[fleet]"
```

