/**
 * Works around two gaps in DigitalOcean's `openhands` sandbox template. Drop
 * this once the template sources the session env and sets TMUX_TMPDIR.
 *
 * - The agent-server starts before the session env (manifest secrets) is
 *   written and never reads it, so it runs without OH_SECRET_KEY and every
 *   encrypted settings read 503s once a secret is stored. The server also
 *   reads `secret_key` from its JSON config file, which lives in the
 *   workspace it owns, so we write one there and restart it once; s6
 *   restarts the process and the key persists with the sandbox disk.
 * - Its TMPDIR pushes tmux's socket path past the Unix limit, so the default
 *   tmux terminal fails as soon as an agent runs. The subprocess terminal
 *   needs no socket.
 *
 * It runs in the bridge when a session is connected, so every caller gets a
 * prepared sandbox: the Canvas page, the desktop app, and an agent that
 * holds the session key alike.
 */

import {
  BashClient,
  ServerClient,
  SettingsClient,
} from "@openhands/typescript-client/clients";
import {
  TUNNEL_CLOSE_GUEST_DIAL_FAILED,
  TUNNEL_CLOSE_SESSION_UNAVAILABLE,
} from "./tunnel-client.mjs";

// The guest agent-server takes no key: the DigitalOcean tunnel is the gate.
const SCRIPT_TIMEOUT_S = 30;
/** The script kills the server one second after answering. */
const RESTART_GRACE_MS = 2_000;

const GUEST_WAIT_MS = 60_000;
const PROBE_TIMEOUT_MS = 5_000;
const PROBE_INTERVAL_MS = 2_000;
/**
 * A just-woken sandbox also closes 4002 while its agent-server boots, so only
 * a 4002 that outlasts this window means nothing will ever serve the port.
 */
const GUEST_PORT_CLOSED_GRACE_MS = 20_000;

/**
 * Prints `restarting` when it wrote a key the running server has not loaded,
 * `ready` when nothing is needed, `unsupported` when the layout is not the
 * one this works around. Generates the key in the guest so it never leaves it.
 */
export const ENSURE_SECRET_KEY_SCRIPT = String.raw`servers=" $(ps -o pid= -C openhands-agent-server | tr -s ' \n' '  ') "
main=""
for p in $servers; do
  parent=$(ps -o ppid= -p "$p" | tr -d ' ')
  case "$servers" in *" $parent "*) ;; *) main=$p ;; esac
done
if [ -z "$main" ]; then echo unsupported; exit 0; fi
if tr '\0' '\n' < "/proc/$main/environ" | grep -q '^OH_SECRET_KEY='; then echo ready; exit 0; fi
cfg=$(tr '\0' '\n' < "/proc/$main/environ" | sed -n 's/^OPENHANDS_AGENT_SERVER_CONFIG_PATH=//p')
cfg=$(cd "/proc/$main/cwd" && realpath -m "$(printf %s "$cfg" | grep . || echo workspace/openhands_agent_server_config.json)")
python3 - "$cfg" <<'PY' || { echo unsupported; exit 0; }
import json, os, secrets, sys
path = sys.argv[1]
data = json.load(open(path)) if os.path.exists(path) else {}
if not data.get("secret_key"):
    data["secret_key"] = secrets.token_hex(32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
PY
started=$(( $(date +%s) - $(ps -o etimes= -p "$main" | tr -d ' ') ))
if [ "$(stat -c %Y "$cfg")" -ge "$started" ]; then
  (sleep 1; kill "$main") >/dev/null 2>&1 &
  echo restarting
else
  echo ready
fi`;

/** The server's default tool set, with the terminal forced off tmux. */
export const MARS_SANDBOX_TOOLS = [
  { name: "terminal", params: { terminal_type: "subprocess" } },
  { name: "file_editor", params: {} },
  { name: "task_tracker", params: {} },
  { name: "browser_tool_set", params: {} },
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * @typedef {{ closeCode: number | null, httpStatus: number | null }} UpstreamFailure
 */

/**
 * Wait until the guest agent-server answers. Gives up early when the tunnel
 * says it never will (refused, or nothing listening on the guest port), so
 * the caller's own checks report that instead of a long wait.
 *
 * @param {object} options
 * @param {string} options.baseUrl The tunnel listener, `http://127.0.0.1:<port>`
 * @param {() => UpstreamFailure | null | undefined} options.readFailure
 * @param {number} [options.budgetMs]
 * @param {number} [options.intervalMs]
 * @returns {Promise<boolean>}
 */
export async function waitForGuest({
  baseUrl,
  readFailure,
  budgetMs = GUEST_WAIT_MS,
  intervalMs = PROBE_INTERVAL_MS,
}) {
  const server = new ServerClient({ host: baseUrl, timeout: PROBE_TIMEOUT_MS });
  const deadline = Date.now() + budgetMs;
  let closedSince = null;
  for (;;) {
    try {
      await server.getServerInfo();
      return true;
    } catch (error) {
      // A guest that answers 401 or 403 will not let us in by waiting;
      // anything else is still waking, or the tunnel still dialing.
      if (error?.status === 401 || error?.status === 403) return false;
    }
    const failure = readFailure() ?? null;
    if (
      failure?.closeCode === TUNNEL_CLOSE_SESSION_UNAVAILABLE ||
      failure?.httpStatus === 401 ||
      failure?.httpStatus === 403
    ) {
      return false;
    }
    const now = Date.now();
    if (failure?.closeCode === TUNNEL_CLOSE_GUEST_DIAL_FAILED) {
      closedSince ??= now;
      if (now - closedSince >= GUEST_PORT_CLOSED_GRACE_MS) return false;
    } else {
      closedSince = null;
    }
    if (now >= deadline) return false;
    await sleep(intervalMs);
  }
}

/**
 * Best-effort: a sandbox this cannot prepare still connects, and the failure
 * surfaces where it bites (conversation start) instead of blocking connect.
 *
 * @param {object} options
 * @param {string} options.baseUrl The tunnel listener, `http://127.0.0.1:<port>`
 * @param {() => UpstreamFailure | null | undefined} options.readFailure
 * @param {(message: string) => void} [options.log]
 * @param {{ budgetMs?: number, intervalMs?: number, restartGraceMs?: number }} [options.timing]
 * @returns {Promise<"prepared" | "unreachable" | "failed">}
 */
export async function prepareSandbox({
  baseUrl,
  readFailure,
  log = (message) => console.warn(`[mars-sandbox] ${message}`),
  timing = {},
}) {
  const wait = () => waitForGuest({ baseUrl, readFailure, ...timing });
  if (!(await wait())) return "unreachable";
  try {
    const output = await new BashClient({ host: baseUrl }).executeCommand({
      command: ENSURE_SECRET_KEY_SCRIPT,
      timeout: SCRIPT_TIMEOUT_S,
    });
    if (output?.stdout?.trim() === "restarting") {
      await sleep(timing.restartGraceMs ?? RESTART_GRACE_MS);
      if (!(await wait())) return "unreachable";
    }
    const settingsClient = new SettingsClient({ host: baseUrl });
    const settings = await settingsClient.getSettings();
    if (settings?.agent_settings?.tools == null) {
      await settingsClient.updateSettings({
        agent_settings_diff: { tools: MARS_SANDBOX_TOOLS },
      });
    }
    return "prepared";
  } catch (error) {
    log(
      `could not prepare the sandbox: ${error instanceof Error ? error.message : String(error)}`,
    );
    return "failed";
  }
}
