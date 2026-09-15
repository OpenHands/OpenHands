/**
 * MARS port-forward tunnel client (MARSOHS-1427).
 *
 * Runs inside Canvas's Electron main process and plays the same role
 * `doctl agents port-forward` plays at the terminal today: given a session
 * id and a remote port inside that session's sandbox, it dials MARS's
 * port-forward WebSocket tunnel and exposes the result as an ordinary local
 * TCP listener that pipes bytes bidirectionally. Rather than reimplementing
 * that WebSocket client in JS, this shells out to the real `doctl` binary
 * (see download-doctl.mjs) — same precedent as bundling uv/uvx.
 *
 * Multi-session note: this module starts one `doctl` process per call, each
 * owning one local listener for one (session, remote port) pair. Building a
 * `session_id -> tunnel` registry on top of this (port allocation, reuse,
 * targeted teardown, reconnect detection) is MARSOHS-1428 — out of scope
 * here.
 */

import { spawn } from "node:child_process";
import net from "node:net";
import path from "node:path";
import process from "node:process";

import {
  getProcessTreeSpawnOptions,
  signalProcessTree,
} from "./dev-process-utils.mjs";

const READY_LINE = "Ready. Press Ctrl-C to stop.";
const DEFAULT_HEALTHY_TIMEOUT_MS = 15_000;

// Matches doctl's `RunAgentsPortForward` announcement, e.g.
// "Forwarding 127.0.0.1:54321 -> port 8000 in session sess_abc123"
const FORWARD_LINE_RE =
  /^Forwarding\s+(\S+):(\d+)\s+->\s+port\s+(\d+)\s+in session\s+(\S+)/;

/**
 * Resolve the doctl binary to spawn: the bundled copy under
 * <resourcesPath>/bin/ when packaged (see electron-builder.config.mjs's
 * extraResources), otherwise whatever `doctl` resolves to on PATH — mirrors
 * main.mjs's injectBundledUv fallback-to-system behavior for dev/tests.
 */
export function resolveDoctlBinary({
  resourcesPath,
  isPackaged,
  platform = process.platform,
} = {}) {
  const binName = platform === "win32" ? "doctl.exe" : "doctl";
  if (isPackaged && resourcesPath) {
    return path.join(resourcesPath, "bin", binName);
  }
  return binName;
}

/**
 * Build the `doctl agents port-forward` argv for one session/port pair.
 * `localPort: 0` lets doctl ask the OS to pick a free port (doctl's own
 * `[<local-port>:]<remote-port>` parsing treats a leading "0:" as such).
 */
export function buildPortForwardArgs({ sessionId, remotePort, localPort = 0 }) {
  if (!sessionId) {
    throw new Error("sessionId is required");
  }
  if (!Number.isInteger(remotePort) || remotePort <= 0 || remotePort > 65535) {
    throw new Error(`Invalid remote port: ${remotePort}`);
  }
  if (!Number.isInteger(localPort) || localPort < 0 || localPort > 65535) {
    throw new Error(`Invalid local port: ${localPort}`);
  }
  return ["agents", "port-forward", sessionId, `${localPort}:${remotePort}`];
}

/** Parse one line of doctl's port-forward stdout, or null if it doesn't match. */
export function parseForwardedPort(line) {
  const match = FORWARD_LINE_RE.exec(line.trim());
  if (!match) return null;
  return {
    address: match[1],
    localPort: Number(match[2]),
    remotePort: Number(match[3]),
    sessionId: match[4],
  };
}

/**
 * Poll a TCP connect against the tunnel's local listener until it accepts a
 * connection or `timeoutMs` elapses. This is the "wait-for-healthy" check
 * before reporting the tunnel ready — the port-forward listener can exist
 * slightly before doctl has finished readying the underlying WS bridge.
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
  doctlPath = "doctl",
  healthyTimeoutMs = DEFAULT_HEALTHY_TIMEOUT_MS,
  spawnFn = spawn,
} = {}) {
  if (!accessToken) {
    throw new Error("accessToken is required");
  }

  const args = buildPortForwardArgs({ sessionId, remotePort, localPort });
  args.push("--access-token", accessToken);
  if (apiUrl) {
    args.push("--api-url", apiUrl);
  }

  const child = spawnFn(
    doctlPath,
    args,
    getProcessTreeSpawnOptions({ stdio: ["ignore", "pipe", "pipe"] }),
  );

  let resolvedPort = null;
  let stderrTail = "";

  const forwardedPort = await new Promise((resolve, reject) => {
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

        const forwarded = parseForwardedPort(line);
        if (forwarded && resolvedPort == null) {
          resolvedPort = forwarded.localPort;
        }
        if (line.trim() === READY_LINE && resolvedPort != null) {
          cleanup();
          resolve(resolvedPort);
          return;
        }
      }
    }

    function onStderr(chunk) {
      // Keep a bounded tail so a hung/rejected dial's error is diagnosable
      // without buffering an unbounded amount of doctl output.
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
          `doctl exited before the tunnel was ready (code=${code ?? "null"}, signal=${signal ?? "null"}): ${stderrTail.trim()}`,
        ),
      );
    }

    child.stdout?.on("data", onStdout);
    child.stderr?.on("data", onStderr);
    child.once("error", onError);
    child.once("exit", onExit);
  });

  await waitForLocalListener(forwardedPort, "127.0.0.1", healthyTimeoutMs);

  return {
    sessionId,
    remotePort,
    localPort: forwardedPort,
    process: child,
    stop(signal = "SIGTERM") {
      return signalProcessTree(child, signal);
    },
  };
}
