# boxd Sandbox Runtime

OpenHands can run agent conversations inside [boxd](https://boxd.sh) cloud
microVMs. Each conversation gets a dedicated KVM-isolated VM with the
OpenHands agent-server image and a public HTTPS proxy on port 60000.

## When to use boxd

- **Docker**: local dev, single machine.
- **Remote**: managed cloud, no infra to run.
- **boxd**: self-hostable cloud (open-source) with per-VM IPv4 and
  sub-millisecond warm suspend/resume. Pause a conversation, return
  a week later, the VM resumes with running processes intact.

## Setup

1. Install the boxd SDK (optional extra):

   ```bash
   pip install boxd
   ```

2. Get a boxd API key (`bxk_...`) from your boxd cluster — see
   https://boxd.sh/docs for cluster setup.

3. Set environment variables before starting the OpenHands server:

   ```bash
   export RUNTIME=boxd
   export BOXD_API_KEY=bxk_...

   # Optional — defaults shown
   export BOXD_AUTO_SUSPEND_TIMEOUT=300   # seconds idle before warm-suspend
   export BOXD_MAX_NUM_SANDBOXES=10       # per-user cap
   export BOXD_VCPU=2                     # per-VM
   export BOXD_MEMORY=8G                  # per-VM
   export BOXD_DISK=100G                  # per-VM
   ```

4. Start OpenHands as normal.

## What happens under the hood

- Each conversation gets a VM named `oh-<sandbox-id>`.
- The agent-server image is pulled by the boxd worker — no local
  Docker pull on the OpenHands host.
- Two HTTPS subdomain proxies are configured at boot:
  - `agent-oh-<sandbox-id>.boxd.sh` → port 60000 (agent server)
  - `vscode-oh-<sandbox-id>.boxd.sh` → port 60001 (VS Code server)
- After `BOXD_AUTO_SUSPEND_TIMEOUT` seconds of inactivity, boxd
  warm-suspends the VM. The next request resumes it in sub-millisecond
  time, preserving running processes and filesystem state.

## Persistence model

A small SQLAlchemy table (`v1_boxd_sandbox`) indexes sandbox metadata
(id, owning user, spec id, session-key hash, created_at). This mirrors
the pattern used by `RemoteSandboxService` and is required because the
boxd SDK doesn't expose environment variables on the returned `Box`
object, so we can't recover metadata directly from the VM.

The live VM state (running / suspended / etc.) is always fetched from
boxd; the DB is the index, not the source of truth for status.

## Security model

- `session_api_key` is generated app-side, injected into the VM via
  the `OH_SESSION_API_KEYS_0` env var, and indexed by SHA-256 hash
  in the app DB. The raw key never persists on the OpenHands host.
- On `pause_sandbox`, the stored hash is cleared so leaked keys can't
  be replayed against the suspended VM until it's resumed.
- `delete_sandbox` drops the index row first, then asks boxd to
  destroy the VM. If boxd is unreachable, the row removal still
  invalidates any leaked keys for that sandbox.

## Known limitations (v1)

- No webhook polling fallback. `RemoteSandboxService` runs a background
  poller against agent servers when the OpenHands `web_url` is on
  localhost; we don't yet. If you run boxd from a localhost OpenHands,
  conversation events from the VM won't reach the app server in real
  time. Provide a public `web_url` (and matching `OH_WEB_URL`) to use
  webhook callbacks.
- The boxd worker pulls the agent-server image lazily on first VM
  creation. The default `wait_for_sandbox_running` timeout is 120
  seconds; bump it for slow image registries.
- Cross-user 404s on `get_sandbox` are deliberate: if you don't own
  the sandbox, you get `None`, not a permission error.

## Troubleshooting

- **`SandboxError: Failed to start sandbox: ...`**: check that
  `BOXD_API_KEY` is set and your boxd cluster is reachable.
- **VM stuck in `STARTING`**: the boxd worker may still be pulling the
  agent-server image. Watch `boxd box logs <vm>` from your local
  `boxd` CLI for image-pull progress.
- **Proxy URL 404**: the agent-server takes ~5–10 seconds to start
  inside the VM. The app server waits via the `/alive` health check.
- **OOM kills**: bump `BOXD_MEMORY`. Default 8G is enough for most
  Python agents but tight for heavy multi-process workloads.
