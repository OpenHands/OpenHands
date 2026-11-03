// @vitest-environment node

import { spawnSync } from "node:child_process";
import { readFileSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const entrypoint = readFileSync(
  path.join(repoRoot, "docker/entrypoint.sh"),
  "utf-8",
);

/** Body of a marker-delimited block, without the marker lines themselves. */
function markerBlock(start: string, end: string): string {
  const from = entrypoint.indexOf(start);
  const to = entrypoint.indexOf(end);
  if (from === -1 || to === -1) {
    throw new Error(`entrypoint is missing the markers ${start} / ${end}`);
  }
  return entrypoint.slice(from + start.length, to);
}

const livenessBlock = markerBlock(
  "# >>> agent-server-liveness",
  "# <<< agent-server-liveness",
);
const exitStatusBlock = markerBlock(
  "# >>> docker-exit-status",
  "# <<< docker-exit-status",
);

function runBash(script: string): {
  status: number | null;
  stdout: string;
  stderr: string;
} {
  const result = spawnSync("bash", ["-c", script], {
    encoding: "utf-8",
    env: { PATH: process.env.PATH ?? "" },
  });
  return {
    status: result.status,
    stdout: result.stdout.trim(),
    stderr: result.stderr,
  };
}

/** True while `pid` is still visible to the process table. */
function isRunning(pid: string): boolean {
  return spawnSync("kill", ["-0", pid], { encoding: "utf-8" }).status === 0;
}

/** Run the liveness block's probe N times against `port` and report each outcome. */
function probeOutcomes(port: number, probes = 3): string[] {
  const script = [
    "set -uo pipefail",
    `AGENT_SERVER_PORT=${port}`,
    "log() { printf '%s\\n' \"$*\" >&2; }",
    "log_error() { printf '%s\\n' \"$*\" >&2; }",
    livenessBlock,
    `for _ in $(seq 1 ${probes}); do`,
    "  if track_agent_server_liveness; then echo alive; else echo dead; fi",
    "done",
  ].join("\n");
  const result = runBash(script);
  expect(result.status).toBe(0);
  return result.stdout ? result.stdout.split("\n") : [];
}

/** Bind and immediately release a port, so nothing is listening on it. */
async function closedPort(): Promise<number> {
  const probeServer = createServer();
  const port = await new Promise<number>((resolve) => {
    probeServer.listen(0, "127.0.0.1", () => {
      const address = probeServer.address();
      if (address && typeof address === "object") resolve(address.port);
    });
  });
  await new Promise<void>((resolve) => probeServer.close(() => resolve()));
  return port;
}

describe("docker entrypoint: agent-server liveness", () => {
  let server: Server | undefined;

  afterEach(async () => {
    if (server) {
      await new Promise<void>((resolve) => server?.close(() => resolve()));
      server = undefined;
    }
  });

  it("keeps reporting the backend as alive while it answers", async () => {
    server = createServer((socket) => socket.end());
    const port = await new Promise<number>((resolve) => {
      server?.listen(0, "127.0.0.1", () => {
        const address = server?.address();
        if (address && typeof address === "object") resolve(address.port);
      });
    });

    expect(probeOutcomes(port)).toEqual(["alive", "alive", "alive"]);
  });

  it("only reports the backend dead once the miss budget is exhausted", async () => {
    // A closed port stands in for a dead agent-server behind a live launcher:
    // PID tracking cannot see the loss, so the probe has to be what notices.
    expect(probeOutcomes(await closedPort())).toEqual([
      "alive",
      "alive",
      "dead",
    ]);
  });

  it("keeps reporting dead for every probe past the budget", async () => {
    const outcomes = probeOutcomes(await closedPort(), 5);
    // The counter resets on every answering probe, so a closed port counts up
    // monotonically and the budget is consumed exactly once.
    expect(outcomes).toEqual(["alive", "alive", "dead", "dead", "dead"]);
  });
});

describe("docker entrypoint: exit status", () => {
  it("carries a failure status out through the EXIT trap", () => {
    // The keep-alive loop exits 1 when the agent-server is gone. Without the
    // status-preserving trap, cleanup()'s own `exit 0` rewrote that into a
    // clean stop and `restart: on-failure` never fired.
    const script = [
      "set -uo pipefail",
      "PIDS=()",
      "log() { :; }",
      exitStatusBlock,
      "exit 1",
    ].join("\n");
    expect(runBash(script).status).toBe(1);
  });

  it("still exits 0 on an intentional clean shutdown", () => {
    const script = [
      "set -uo pipefail",
      "PIDS=()",
      "log() { :; }",
      exitStatusBlock,
      "exit 0",
    ].join("\n");
    expect(runBash(script).status).toBe(0);
  });

  it("stops the tracked child processes before exiting", () => {
    // The failing path has to take the other services down with it, otherwise a
    // restart leaves the old children behind.
    const pidFile = path.join(tmpdir(), `entrypoint-child-${process.pid}.pid`);
    const script = [
      "set -uo pipefail",
      "PIDS=()",
      "log() { :; }",
      exitStatusBlock,
      `sleep 30 & PIDS+=($!); printf '%s' "$!" > "${pidFile}"`,
      "exit 1",
    ].join("\n");

    expect(runBash(script).status).toBe(1);

    const childPid = readFileSync(pidFile, "utf-8").trim();
    rmSync(pidFile, { force: true });
    expect(isRunning(childPid)).toBe(false);
  });
});
