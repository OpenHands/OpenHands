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
 */

import {
  BashClient,
  SettingsClient,
} from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import {
  MARS_GUEST_API_KEY,
  waitForMarsAgentServer,
} from "#/api/mars/mars-tunnel-backend";

const SETUP_TIMEOUT_MS = 30_000;
/** The script kills the server one second after answering. */
const RESTART_GRACE_MS = 2_000;

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

/**
 * Best-effort: a sandbox this cannot prepare still connects, and the failure
 * surfaces where it bites (conversation start) instead of blocking attach.
 */
export async function prepareMarsSandbox(
  host: string,
  sessionId: string,
): Promise<void> {
  const options = getAgentServerClientOptions({
    host,
    sessionApiKey: MARS_GUEST_API_KEY,
    timeout: SETUP_TIMEOUT_MS,
  });
  try {
    const output = await new BashClient(options).executeCommand({
      command: ENSURE_SECRET_KEY_SCRIPT,
      timeout: SETUP_TIMEOUT_MS / 1000,
    });
    if (output.stdout?.trim() === "restarting") {
      await new Promise((resolve) => {
        setTimeout(resolve, RESTART_GRACE_MS);
      });
      await waitForMarsAgentServer(host, { sessionId });
    }

    const settingsClient = new SettingsClient(options);
    const settings = await settingsClient.getSettings();
    if (settings.agent_settings?.tools == null) {
      await settingsClient.updateSettings({
        agent_settings_diff: { tools: MARS_SANDBOX_TOOLS },
      });
    }
  } catch (error) {
    console.warn("Could not prepare the DigitalOcean sandbox", error);
  }
}
