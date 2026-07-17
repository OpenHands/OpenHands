# Operating the Rok1375 OpenHands Fork

This guide is for the checked-out monorepo source in this fork. Upstream OpenHands is transitioning Agent Canvas and the agent SDK into separate repositories, but this operator intentionally uses the build and run targets that exist in this fork.

The fork includes a dependency-free operator CLI that checks the machine, prepares local state, validates provider configuration, and launches the existing OpenHands source stack without changing upstream backend or frontend behavior.

## Security model

OpenHands is an autonomous coding agent. Treat every workspace mount and credential as privileged access.

- Use the Docker runtime unless you deliberately need local execution.
- Local runtime can access the host filesystem with the permissions of the account running OpenHands.
- The operator refuses non-loopback binding unless `--allow-remote-access` is supplied.
- That flag is an acknowledgement, not a security control. Put remote deployments behind TLS, authentication, firewall rules, and a hardened reverse proxy.
- Never commit API keys to this repository. `.env`, `config.toml`, and the default workspace are ignored, but environment or secret-store injection is still preferred.
- ChatGPT/Codex browser login, OpenAI API authentication, OpenHands application login, and third-party provider credentials are separate systems. Do not copy browser cookies or session tokens into model settings.

## Prerequisites

The `doctor` command validates the source-development baseline used by this repository:

- Python `>=3.12,<3.14`
- Node.js `>=22.12.0`
- npm
- Poetry `>=1.8`
- Git
- GNU Make
- netcat (`nc`), which the existing `make run` readiness loop requires
- Docker CLI and a reachable Docker daemon when `--runtime docker` is selected
- writable workspace
- valid TCP ports and available backend/frontend bindings

Run the full check from the repository root:

```bash
python scripts/openhands_operator.py doctor
```

A successful report ends with zero errors. Warnings are non-blocking unless `--strict` is supplied.

## Recommended first start

Docker remains the safer runtime for agent execution:

```bash
python scripts/openhands_operator.py start \
  --runtime docker \
  --provider auto \
  --bootstrap \
  --build
```

This command:

1. creates `./workspace` when absent;
2. validates the host, Docker daemon, workspace, ports, remote-binding policy, and provider environment;
3. runs `make build`;
4. starts the existing OpenHands backend and frontend with `make run`;
5. passes `RUNTIME=docker` and the resolved workspace to the child process.

When no environment provider is configured, startup reports a warning and continues so the model can be configured in the V1 Settings UI.

Open the local frontend at `http://127.0.0.1:3001` unless different ports were selected.

## Preview before executing

Use a dry run to inspect the commands and non-sensitive environment summary:

```bash
python scripts/openhands_operator.py start \
  --runtime docker \
  --provider auto \
  --bootstrap \
  --build \
  --dry-run
```

Dry-run mode does not create the workspace, copy `config.toml`, run the build, or launch the application. The operator reports only whether an API key or base URL is set and never prints an API-key value.

## Provider configuration

### Configure in the OpenHands UI

No provider variables are required when you plan to configure the model after startup:

```bash
python scripts/openhands_operator.py start --bootstrap --build
```

Use `--require-provider` when unattended startup must fail unless environment-based model credentials are complete.

### Generic LiteLLM/OpenAI-compatible provider

Set the existing OpenHands variables in the shell or a secret manager:

```bash
export LLM_MODEL='<provider>/<model-id>'
export LLM_API_KEY='<secret>'
# Optional for providers that need a custom OpenAI-compatible endpoint:
export LLM_BASE_URL='<https://provider.example/v1>'

python scripts/openhands_operator.py doctor \
  --provider generic \
  --require-provider

python scripts/openhands_operator.py start \
  --provider generic \
  --require-provider \
  --bootstrap \
  --build
```

When `LLM_BASE_URL` is set, it must be an absolute `http://` or `https://` URL and must not contain embedded username/password credentials.

Confirm the current model identifier and endpoint in the provider's official documentation or account dashboard. Do not guess them or hard-code them into this repository.

### OpenCode Go profile

Set these fork-specific variables through the shell, GitHub Codespaces secrets, or a service environment file:

```bash
export OPENCODE_GO_MODEL='<confirmed-model-id>'
export OPENCODE_GO_BASE_URL='<confirmed-openai-compatible-base-url>'
export OPENCODE_GO_API_KEY='<secret>'
```

Validate and start:

```bash
python scripts/openhands_operator.py doctor \
  --provider opencode-go \
  --require-provider

python scripts/openhands_operator.py start \
  --provider opencode-go \
  --require-provider \
  --bootstrap \
  --build
```

For the launched child process only, the operator maps the profile into:

```text
LLM_MODEL=openai/<OPENCODE_GO_MODEL>
LLM_BASE_URL=<OPENCODE_GO_BASE_URL>
LLM_API_KEY=<OPENCODE_GO_API_KEY>
```

A model value that already begins with `openai/` is preserved. Other values, including IDs that contain a slash, receive the `openai/` compatibility prefix.

## Local runtime

Use local runtime only in a trusted disposable machine or workspace:

```bash
python scripts/openhands_operator.py start \
  --runtime local \
  --provider auto \
  --bootstrap \
  --build
```

The child process receives `INSTALL_DOCKER=0`, which prevents the source build from treating Docker as a prerequisite. The readiness report warns that the agent may access the host filesystem directly.

## WSL 2

Keep the repository and workspace inside the Linux filesystem, for example `~/src/OpenHands`, rather than `/mnt/c`. Start Docker Desktop with WSL integration before choosing the Docker runtime.

```bash
cd ~/src/OpenHands
python scripts/openhands_operator.py doctor --runtime docker
python scripts/openhands_operator.py start --runtime docker --bootstrap --build
```

If Docker is intentionally unavailable, choose `--runtime local` and review the filesystem-access warning before continuing.

## GitHub Codespaces

Store provider keys as Codespaces secrets, rebuild or restart the Codespace after adding them, and verify only their presence:

```bash
if [ -n "${OPENCODE_GO_API_KEY:-}" ]; then
  echo 'OpenCode Go key is available'
else
  echo 'OpenCode Go key is missing' >&2
  exit 1
fi
```

Then run:

```bash
python scripts/openhands_operator.py doctor \
  --provider opencode-go \
  --require-provider
```

Use Docker runtime only when the Codespace has a reachable Docker daemon. Otherwise, use local runtime deliberately and do not mount unrelated private directories into the workspace.

Forward only the frontend port you need. Keep forwarded ports private unless the deployment has an independent authentication and access-control boundary.

## Private VPS or remote development host

The secure default binds only to loopback. Put a TLS reverse proxy or SSH tunnel in front of it.

A direct non-loopback bind is blocked until explicitly acknowledged:

```bash
python scripts/openhands_operator.py start \
  --runtime docker \
  --bootstrap \
  --build \
  --backend-host 0.0.0.0 \
  --frontend-host 0.0.0.0 \
  --allow-remote-access
```

Before using this mode, provide all of the following:

- TLS termination;
- authentication or an identity-aware proxy;
- host firewall rules;
- restricted workspace mounts;
- non-root service account where practical;
- protected secret injection;
- log rotation and disk monitoring;
- backups for persistent OpenHands state.

The repository's `make run` path is a source-development server. Do not expose it directly to the public internet or describe it as a hardened production deployment.

## Bootstrap commands

Create only the workspace:

```bash
python scripts/openhands_operator.py bootstrap
```

Create the workspace and an ignored local `config.toml` copied from the repository template:

```bash
python scripts/openhands_operator.py bootstrap --create-config
```

An existing `config.toml` is never overwritten.

## JSON and CI mode

Machine-readable readiness report:

```bash
python scripts/openhands_operator.py doctor --json
```

Make missing provider configuration blocking:

```bash
python scripts/openhands_operator.py doctor \
  --json \
  --require-provider
```

Make every warning blocking, including local-runtime and missing-workspace warnings:

```bash
python scripts/openhands_operator.py doctor --json --strict
```

`--skip-system-checks` and `--skip-port-checks` exist for focused tests and constrained CI jobs. Do not use them as a normal way to bypass a failing production-readiness check.

## Troubleshooting

### Docker daemon is not reachable

Start Docker Desktop or Docker Engine, confirm `docker info` succeeds, and rerun `doctor`. Choose local runtime only when its reduced isolation is acceptable.

### A prerequisite is missing

Install the exact command named in the report. In particular, `make run` waits for the backend with `nc`, so a host without netcat will be blocked before launch.

### A port is invalid or occupied

Ports must be between `1` and `65535`. Select unused ports when the defaults are occupied:

```bash
python scripts/openhands_operator.py start \
  --backend-port 3100 \
  --frontend-port 3101 \
  --bootstrap
```

### Provider configuration is incomplete

The report lists missing variable names, never their values. Set every required variable for the selected profile or choose `--provider auto` and configure the model in the Settings UI.

### Build or launch exits early

The operator returns the exit status from `make build` or `make run`. Resolve the first failing build message, then rerun the same command. For deeper source-development guidance, see the repository `Development.md` and root `AGENTS.md`.

## Verification for contributors

The operator layer is intentionally dependency-free:

```bash
python -m py_compile scripts/openhands_operator.py
python -m unittest tests.unit.test_openhands_operator -v
python scripts/openhands_operator.py --help
```

The dedicated `Fork operator readiness` GitHub Actions workflow runs the same checks when these files change.
