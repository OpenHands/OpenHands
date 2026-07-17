# OpenHands Fork Operator Readiness Design

## Objective

Make this fork safe and predictable to start in local development, Codespaces, WSL, and a private VPS without changing OpenHands backend/frontend APIs or introducing a hard fork.

The feature is intentionally an operator layer rather than a rewrite of authentication, model, frontend, or runtime internals. OpenHands is moving quickly upstream, so the safest long-term enhancement is a tested readiness and launch workflow that remains compatible with upstream merges. One existing Makefile wiring defect is corrected so `FRONTEND_HOST` actually controls the Vite bind address.

## Success Criteria

1. A single command can prepare the workspace, validate the host, optionally build, and start OpenHands.
2. Readiness failures identify the exact missing or incompatible prerequisite.
3. Docker and local runtime modes are both supported, with Docker remaining the safer default.
4. Generic OpenAI-compatible and OpenCode Go environment configurations are validated without printing secrets.
5. OpenCode Go variables can be mapped into the existing `LLM_*` environment expected by OpenHands.
6. Missing model credentials remain a warning by default because the V1 web UI can configure them after startup; operators may make them mandatory with a flag.
7. Human-readable and JSON reports are available.
8. The operator code uses only the Python standard library and has focused unit tests.
9. A lightweight GitHub Actions workflow blocks regressions in the operator layer and frontend host wiring.
10. Documentation covers WSL/Codespaces/VPS startup, provider setup, remote-access risks, and the separation between ChatGPT browser login and API credentials.
11. Backend and frontend hosts can be selected independently and are honored by the existing Make targets.

## Architecture

### `scripts/openhands_operator.py`

A standard-library Python CLI with three subcommands:

- `doctor`: returns a readiness report and exit status.
- `bootstrap`: creates the workspace and optionally copies `config.template.toml` to the ignored `config.toml` path.
- `start`: optionally bootstraps and builds, runs the same readiness checks, safely maps provider variables, then launches `make run`.

The CLI is split into pure helpers for version parsing, provider validation, report rendering, command construction, and environment mapping. System probes are isolated behind small functions so tests can replace them.

### `Makefile`

The existing `start-frontend` target is corrected to pass `FRONTEND_HOST` to Vite rather than reusing `BACKEND_HOST`. This is a one-line operational bug fix; backend/frontend application APIs and implementation remain unchanged.

### Readiness checks

- Python version: `>=3.12,<3.14`.
- Node.js version: `>=22.12.0`.
- npm, Poetry, Git, GNU Make, and netcat (`nc`) availability.
- Poetry version: `>=1.8`.
- Docker CLI and daemon when runtime is `docker`.
- Workspace existence and writability.
- Backend and frontend port validity and availability unless disabled.
- Loopback/non-loopback binding safety.
- Provider configuration completeness and custom base-URL validity.

Checks are classified as `pass`, `warning`, or `error`. Exit code is `1` only when at least one error exists. `--strict` upgrades warnings to errors. `--require-provider` makes missing provider configuration an error without making unrelated informational warnings fatal.

### Provider behavior

Provider mode is `auto`, `generic`, `opencode-go`, or `none`.

- `generic` requires `LLM_MODEL` and `LLM_API_KEY`; `LLM_BASE_URL` is optional.
- `opencode-go` requires `OPENCODE_GO_MODEL`, `OPENCODE_GO_BASE_URL`, and `OPENCODE_GO_API_KEY`.
- `auto` selects OpenCode Go when any `OPENCODE_GO_*` variable is present, otherwise generic when any `LLM_*` variable is present, otherwise reports that the Settings UI may be used after startup.
- `none` skips provider validation unless `--require-provider` is also supplied.

When OpenCode Go is complete, `start` maps it only in the child process:

- `LLM_MODEL=openai/<OPENCODE_GO_MODEL>` unless the model already starts with `openai/`.
- `LLM_BASE_URL=<OPENCODE_GO_BASE_URL>`.
- `LLM_API_KEY=<OPENCODE_GO_API_KEY>`.

No report, command preview, exception, or documentation prints a secret value. Custom base URLs must be absolute HTTP(S) URLs and cannot embed credentials.

## Data Flow

1. Parse CLI arguments and environment.
2. Resolve runtime, workspace, validated ports, hosts, and provider mode.
3. Run system, workspace, binding, port, and provider probes.
4. Render human or JSON output.
5. Stop on errors.
6. For `start`, optionally create local files/directories, optionally run `make build`, build a sanitized launch description, and execute `make run` with the child environment.
7. In dry-run mode, report the intended bootstrap/build/start actions without mutating files or launching commands.

## Error Handling

- Missing commands produce actionable errors rather than tracebacks.
- Invalid version output produces an error naming the command.
- Invalid TCP ports fail during argument parsing.
- Partial provider configuration lists missing variable names but never populated values.
- Invalid provider URLs name the variable but do not echo the supplied value.
- Docker daemon failures explain that Docker Desktop/Engine must be running or that `--runtime local` can be chosen deliberately.
- A missing workspace is a warning for `doctor` and is created by `bootstrap` or non-dry-run `start --bootstrap`.
- Port conflicts name the occupied port and can be bypassed only with `--skip-port-checks`.
- Non-loopback binding is refused unless the operator explicitly supplies `--allow-remote-access`.
- Interrupted builds or launches return the child process exit code.

## Testing

`tests/unit/test_openhands_operator.py` uses `unittest` and mocks probes so it runs without installing OpenHands dependencies.

Coverage includes:

- semantic version and TCP-port parsing;
- partial and complete generic/OpenCode Go configuration;
- provider auto-detection;
- OpenCode Go child-environment mapping, including slash-containing model IDs;
- provider base-URL validation;
- secret redaction in human and JSON reports;
- workspace/config bootstrap behavior;
- non-mutating sanitized dry-run output;
- netcat prerequisite detection;
- remote-binding acknowledgement;
- error exit behavior.

`.github/workflows/fork-operator-readiness.yml` runs syntax compilation, the unit suite, CLI help, and an exact Makefile frontend-host wiring check on relevant changes.

## Scope Boundaries

- No browser-session credential scraping or conversion.
- No new ChatGPT authentication implementation.
- No hard-coded provider endpoint, model identifier, or key.
- No modifications to upstream backend/frontend APIs.
- No public-internet deployment automation that bypasses TLS, access control, or reverse-proxy hardening.
- No dependency or lockfile changes.
- No application source changes beyond the one-line Makefile host-variable correction.

## Rollback

Remove the operator script, tests, workflow, and fork documentation, then restore the `start-frontend` Vite host argument from `FRONTEND_HOST` to its previous value. No backend/frontend API or persistent-data migration is involved.
