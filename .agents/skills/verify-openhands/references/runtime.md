# Isolated runtime recipe (Linux/macOS)

Run from the repository root in a dedicated shell. Require Node **>=24**, npm,
uv/uvx, git and a Playwright-compatible browser. Read `package.json`,
`config/defaults.json`, and [DEVELOPMENT.md](../../../../docs/DEVELOPMENT.md) at the
revision under test. Use the checked-in lockfile and version pins, not a guessed
"latest" SDK. A Windows/native install is a separate matrix cell.

## Build the exact checkout

Use a clean, dedicated worktree without a `.env` file or inherited provider keys.
Install dependencies before the isolated runtime environment is created:

```sh
npm ci --ignore-scripts --no-audit --no-fund
VITE_DO_NOT_TRACK=1 npm run build:app && git rev-parse HEAD > build/verify-revision.txt
```

The lifecycle-script-free install avoids unrelated desktop/Husky setup; the web
build runs its translation step explicitly. Disable telemetry at build time too:
runtime environment flags do not rewrite a precompiled Vite bundle. If this stops working, record the
actual missing prerequisite rather than enabling arbitrary install hooks.
Use an already installed Chromium through `QA_BROWSER_EXECUTABLE`, or install the
pinned Playwright browser with `npx playwright install chromium` when authorized.

## Allocate a run and start the existing production launcher

Do not use the public forwarded ports of a hosted workspace. Choose free local
ports; if the launcher selects alternatives, update the recorded URLs and doctor
them before driving. Never kill another listener to obtain a preferred port.
The launcher can bind frontend/ingress beyond loopback: **do not expose these
ports externally**. Public mode below keeps the session key out of served HTML;
it is an authentication mode, not permission to publish this instance.

```sh
export QA_RUN="$(mktemp -d "${TMPDIR:-/tmp}/canvas-verify.XXXXXX")"
export QA_BASE_URL=http://127.0.0.1:18800
export QA_AGENT_URL=http://127.0.0.1:18801
export QA_AUTOMATION_URL=http://127.0.0.1:18802
mkdir -p "$QA_RUN/private/home" "$QA_RUN/private/state/canvas" \
  "$QA_RUN/private/workspace" "$QA_RUN/evidence"
node --input-type=module <<'JS'
import { randomBytes } from 'node:crypto';
import { writeFileSync } from 'node:fs';
for (const name of ['session-key', 'encryption-key']) {
  writeFileSync(`${process.env.QA_RUN}/private/${name}`, randomBytes(32).toString('hex'), { mode: 0o600 });
}
JS

env -i PATH="$PATH" HOME="$QA_RUN/private/home" LANG="${LANG:-C.UTF-8}" \
  UV_CACHE_DIR="$HOME/.cache/uv" UV_PYTHON=3.12 \
  OH_CANVAS_SAFE_STATE_DIR="$QA_RUN/private/state/canvas" \
  OH_CANVAS_SAFE_BACKEND_PORT=18801 OH_CANVAS_SAFE_AUTOMATION_PORT=18802 \
  OH_CANVAS_SAFE_VITE_PORT=18803 OH_CANVAS_SAFE_VSCODE_PORT=18804 \
  VITE_WORKING_DIR="$QA_RUN/private/workspace" \
  LOCAL_BACKEND_API_KEY="$(cat "$QA_RUN/private/session-key")" \
  OH_SECRET_KEY="$(cat "$QA_RUN/private/encryption-key")" \
  VITE_DO_NOT_TRACK=1 DO_NOT_TRACK=1 \
  node bin/agent-canvas.mjs --public --port 18800 \
  > "$QA_RUN/private/stack.log" 2>&1 &
echo "$!" > "$QA_RUN/private/launcher.pid"
```

Keep that shell and its environment for doctor/drive/cleanup. No real provider
keys are inherited. Use the UI to configure a separately authorized test profile
when LLM execution is in scope. Public-mode first-run onboarding can be skipped
through `onboarding-skip` to reach `api-key-entry-screen`; use its name/key fields
and submit button. Skipping onboarding is **not** an onboarding-success pass.

Store the build SHA, launcher PID, actual ports, start time, and versions in run
notes. Inspect the private log for the ready banner; uv's first launch can take
several minutes. Do not dump that log publicly: it may contain private state.

## Doctor (read-only)

Check the saved PID with `ps -p "$(cat "$QA_RUN/private/launcher.pid")"` and inspect
its descendants/listeners before assuming port ownership. Then run:

```sh
node --input-type=module <<'JS'
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
const key = readFileSync(`${process.env.QA_RUN}/private/session-key`, 'utf8').trim();
const expected = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const get = (url, auth = false) => fetch(url, {
  redirect: 'error',
  headers: auth ? { 'X-Session-API-Key': key } : {},
  signal: AbortSignal.timeout(10000),
});
const marker = await get(`${process.env.QA_BASE_URL}/verify-revision.txt`);
assert.equal(marker.status, 200);
assert.equal((await marker.text()).trim(), expected);
const unauth = await get(`${process.env.QA_BASE_URL}/api/settings`);
assert.ok([401, 403].includes(unauth.status));
const settings = await get(`${process.env.QA_BASE_URL}/api/settings`, true);
assert.equal(settings.status, 200);
await settings.json();
const info = await get(`${process.env.QA_BASE_URL}/server_info`, true);
assert.equal(info.status, 200);
const server = await info.json();
const automation = await get(`${process.env.QA_BASE_URL}/api/automation/health`, true);
assert.equal(automation.status, 200);
await automation.json();
console.log(JSON.stringify({ build: expected, agentServer: server.version, authenticated: true, automationHealthy: true }));
JS
```

Check `config/defaults.json` against actual versions. Read the automation package
version from its startup/install metadata; `/api/automation/sdk-version` reports
the SDK it uses, not the automation package version. A mismatch is a recorded
prerequisite difference, not something to conceal. Re-run doctor after failures.
The build marker is added only after a successful build; do not reuse a stale
build and rewrite its marker to make this check pass.

## Drive and retain proof

Run the smoke helper from the skill, then follow the mapped user recipes. Do not
point it at a shared/production service. Its unique dummy secret is the only
persisted test fixture it creates; it removes it before returning. If the helper
fails mid-mutation, record the `QA_VERIFY_*` fixture name from its private results
and remove that fixture in the run's UI before teardown. The isolated state also
ensures a failed attempt cannot affect the user's real settings.

For baseline comparison, stop this instance before reusing the same ports, or
allocate a completely distinct port set and run root. Keep separate browser
contexts and evidence directories, labeled by full revision and feature ID.

## Cleanup without deleting evidence

Close the browser context. Disable/remove run-owned automation fixtures and any
other integrations or remote resources explicitly created during this run.
Validate the saved PID still belongs to this launcher (avoid PID reuse), then:

```sh
kill -TERM "$(cat "$QA_RUN/private/launcher.pid")"
```

The launcher owns its child-process shutdown. Inspect its children and recorded
ports until all owned services exit. If a child remains, confirm its identity and
terminate that specific PID, never `pkill -f node` or a similar broad pattern.
Do not delete state while a process still uses it.

Keep `$QA_RUN/evidence` and the run ledger. Remove `$QA_RUN/private` only after
checking the canonical run path, that it is not a symlink, and that teardown is
complete. Verify the evidence files still exist after cleanup. Do not remove the
whole run directory or publish it wholesale. Persist reviewed evidence to the
operator's approved durable destination before the environment can be recycled.
