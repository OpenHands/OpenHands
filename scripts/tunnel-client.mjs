/**
 * MARS port-forward tunnel client (MARSOHS-1427).
 *
 * Runs inside Canvas's Electron main process and plays the same role
 * `doctl agents port-forward` plays at the terminal today: given a session
 * id and a remote port inside that session's sandbox, it dials MARS's
 * port-forward WebSocket tunnel and exposes the result as an ordinary local
 * TCP listener that pipes bytes bidirectionally.
 *
 * v1 implementation language is Python (tools/mars_tunnel.py), a port of
 * doctl's `agent_port_forward.go` reference implementation, run via `uv run`
 * — the same bundled Python runtime uv/uvx already provide for the Agent
 * Server itself. This adds no new runtime dependency and no dependency on a
 * doctl release (see openhands-canvas-dataplane-design.md's "Decision"
 * section for why this replaced an earlier bundled-`doctl`-binary approach).
 *
 * Multi-session note: this module starts one `uv run` subprocess per call,
 * each owning one local listener for one (session, remote port) pair.
 * Building a `session_id -> tunnel` registry on top of this (port
 * allocation, reuse, targeted teardown, reconnect detection) is a separate,
 * follow-on ticket — out of scope here.
 */

import { spawn } from "node:child_process";
import net from "node:net";
import { fileURLToPath } from "node:url";

import {
  getProcessTreeSpawnOptions,
  signalProcessTree,
} from "./dev-process-utils.mjs";

const DEFAULT_HEALTHY_TIMEOUT_MS = 15_000;

// tools/ is a sibling of scripts/ both in dev (repo root) and packaged
// (Resources/app/) layouts — see dev-safe.mjs's canvasToolsDir for the same
// pattern, used there to locate tools/canvas_ui_tool.py.
const MARS_TUNNEL_SCRIPT_PATH = fileURLToPath(
  new URL("../tools/mars_tunnel.py", import.meta.url),
);

/** Matches mars_tunnel.py's `TUNNEL_READY <port>` readiness line. */
const READY_LINE_RE = /^TUNNEL_READY\s+(\d+)\s*$/;

export function resolveMarsTunnelScriptPath() {
  return MARS_TUNNEL_SCRIPT_PATH;
}

/**
 * Parse one line of the tunnel subprocess's stdout, or null if it isn't the
 * readiness line.
 */
export function parseTunnelReadyLine(line) {
  const match = READY_LINE_RE.exec(line.trim());
  if (!match) return null;
  return Number(match[1]);
}

/**
 * Poll a TCP connect against the tunnel's local listener until it accepts a
 * connection or `timeoutMs` elapses. This is the "wait-for-healthy" check
 * before reporting the tunnel ready (mirrors dev-extra-backend.mjs's
 * waitForServer pattern) — independent of the subprocess's own readiness
 * claim, which only confirms the listener socket was bound, not that a
 * client can actually reach it.
 */
function waitForLocalListener(port, host, timeoutMs) {
  const deadline = Date.now() + timeoutMs;

  return new Promise((resolve, reject) => {
    const attempt = () => {
      const socket = net.connect({ port, host }, () => {
        socket.end();
        resolve();
      });
      socket.once("error", () => {
        socket.destroy();
        if (Date.now() >= deadline) {
          reject(
            new Error(
              `Timed out waiting for tunnel listener on ${host}:${port} (${timeoutMs}ms)`,
            ),
          );
          return;
        }
        setTimeout(attempt, 200);
      });
    };
    attempt();
  });
}

/**
 * Dial the MARS port-forward tunnel for one session/port pair and resolve
 * once the local listener is up and accepting connections.
 *
 * @returns {Promise<{sessionId: string, remotePort: number, localPort: number, process: import('node:child_process').ChildProcess, stop: (signal?: string) => boolean}>}
 */
export async function startPortForwardTunnel({
  sessionId,
  remotePort,
  accessToken,
  apiUrl,
  localPort = 0,
  address = "127.0.0.1",
  scriptPath = MARS_TUNNEL_SCRIPT_PATH,
  uvCommand = "uv",
  healthyTimeoutMs = DEFAULT_HEALTHY_TIMEOUT_MS,
  spawnFn = spawn,
} = {}) {
  if (!sessionId) {
    throw new Error("sessionId is required");
  }
  if (!Number.isInteger(remotePort) || remotePort <= 0 || remotePort > 65535) {
    throw new Error(`Invalid remote port: ${remotePort}`);
  }
  if (!accessToken) {
    throw new Error("accessToken is required");
  }

  const args = [
    "run",
    scriptPath,
    "--session-id",
    sessionId,
    "--remote-port",
    String(remotePort),
    "--local-port",
    String(localPort),
    "--address",
    address,
  ];
  if (apiUrl) {
    args.push("--api-url", apiUrl);
  }

  const child = spawnFn(
    uvCommand,
    args,
    getProcessTreeSpawnOptions({
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        // Passed via env, not argv, so the bearer token never shows up in a
        // `ps` listing (mars_tunnel.py reads this exact variable).
        MARS_TUNNEL_ACCESS_TOKEN: accessToken,
      },
    }),
  );

  let stderrTail = "";

  const announcedPort = await new Promise((resolve, reject) => {
    let stdoutBuffer = "";

    function cleanup() {
      child.stdout?.off("data", onStdout);
      child.stderr?.off("data", onStderr);
      child.off("error", onError);
      child.off("exit", onExit);
    }

    function onStdout(chunk) {
      stdoutBuffer += chunk.toString();
      let newlineIndex;
      while ((newlineIndex = stdoutBuffer.indexOf("\n")) !== -1) {
        const line = stdoutBuffer.slice(0, newlineIndex);
        stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);

        const readyPort = parseTunnelReadyLine(line);
        if (readyPort != null) {
          cleanup();
          resolve(readyPort);
          return;
        }
      }
    }

    function onStderr(chunk) {
      // Keep a bounded tail so a hung/rejected dial's error is diagnosable
      // without buffering an unbounded amount of subprocess output.
      stderrTail = (stderrTail + chunk.toString()).slice(-4000);
    }

    function onError(error) {
      cleanup();
      reject(error);
    }

    function onExit(code, signal) {
      cleanup();
      reject(
        new Error(
          `mars_tunnel.py exited before the tunnel was ready (code=${code ?? "null"}, signal=${signal ?? "null"}): ${stderrTail.trim()}`,
        ),
      );
    }

    child.stdout?.on("data", onStdout);
    child.stderr?.on("data", onStderr);
    child.once("error", onError);
    child.once("exit", onExit);
  });

  await waitForLocalListener(announcedPort, "127.0.0.1", healthyTimeoutMs);

  return {
    sessionId,
    remotePort,
    localPort: announcedPort,
    process: child,
    stop(signal = "SIGTERM") {
      return signalProcessTree(child, signal);
    },
  };
}
