// @vitest-environment node

import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import {
  createServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

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

/**
 * Run a bash script without blocking the event loop, so HTTP servers started
 * by the test can answer the probe while it runs. A death by signal reports -1.
 */
function runBash(
  script: string,
  env: Record<string, string> = {},
): Promise<{ status: number; stdout: string }> {
  return new Promise((resolve) => {
    const child = execFile(
      "bash",
      ["-c", script],
      { encoding: "utf-8", env: { PATH: process.env.PATH ?? "", ...env } },
      (_error, stdout) => {
        resolve({ status: child.exitCode ?? -1, stdout: stdout.trim() });
      },
    );
  });
}

async function listen(server: Server): Promise<number> {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (address && typeof address === "object") resolve(address.port);
    });
  });
}

/** Bind and immediately release a port, so nothing is listening on it. */
async function closedPort(): Promise<number> {
  const probeServer = createServer();
  const port = await listen(probeServer);
  await new Promise<void>((resolve) => probeServer.close(() => resolve()));
  return port;
}

/**
 * Run the liveness tracker `probes` times against `port` and report each
 * outcome. `upAtStart` mirrors the startup wait having seen the port open.
 */
async function probeOutcomes(
  port: number,
  probes: number,
  {
    env = {},
    upAtStart = false,
  }: {
    env?: Record<string, string>;
    upAtStart?: boolean;
  } = {},
): Promise<string[]> {
  const script = [
    "set -uo pipefail",
    `AGENT_SERVER_PORT=${port}`,
    "AGENT_SERVER_PROBE_TIMEOUT=0.5",
    "log() { :; }",
    "log_error() { :; }",
    livenessBlock,
    upAtStart ? "mark_agent_server_up" : "",
    `for _ in $(seq 1 ${probes}); do`,
    "  if track_agent_server_liveness; then echo alive; else echo dead; fi",
    "done",
  ].join("\n");
  const result = await runBash(script, env);
  expect(result.status).toBe(0);
  return result.stdout ? result.stdout.split("\n") : [];
}

describe("docker entrypoint: agent-server liveness", () => {
  let server: Server | undefined;

  /** Serve /alive once, then hand every later request to `afterFirst`. */
  async function answerOnceThen(
    afterFirst: (request: IncomingMessage, response: ServerResponse) => void,
  ): Promise<number> {
    let answered = false;
    server = createServer((request, response) => {
      if (answered) {
        afterFirst(request, response);
        return;
      }
      answered = true;
      response.end();
    });
    return listen(server);
  }

  afterEach(async () => {
    if (server) {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server?.close(() => resolve()));
      server = undefined;
    }
  });

  it("probes /alive directly while the backend answers, even with a proxy configured", async () => {
    const paths: string[] = [];
    server = createServer((request, response) => {
      paths.push(request.url ?? "");
      response.end();
    });
    const port = await listen(server);

    const outcomes = await probeOutcomes(port, 3, {
      env: { http_proxy: "http://127.0.0.1:9" },
    });

    expect(outcomes).toEqual(["alive", "alive", "alive"]);
    expect(paths).toEqual(["/alive", "/alive", "/alive"]);
  });

  it("reports the backend dead once three probes in a row miss after it has answered", async () => {
    // A dropped connection stands in for a dead agent-server behind a live
    // launcher: PID tracking cannot see the loss, so the probe has to.
    const port = await answerOnceThen((request) => request.socket.destroy());

    expect(await probeOutcomes(port, 4)).toEqual([
      "alive",
      "alive",
      "alive",
      "dead",
    ]);
  });

  it("counts a server that accepts connections but never answers as a miss", async () => {
    // A hung event loop still completes the TCP handshake, so only an HTTP
    // answer within the timeout proves the server is serving.
    const port = await answerOnceThen(() => {});

    expect(await probeOutcomes(port, 4)).toEqual([
      "alive",
      "alive",
      "alive",
      "dead",
    ]);
  }, 15_000);

  it("keeps waiting for a backend that has not come up yet", async () => {
    // A first start slower than the startup wait is not a crash, so misses
    // before the backend first answers do not count against the budget.
    expect(await probeOutcomes(await closedPort(), 5)).toEqual([
      "alive",
      "alive",
      "alive",
      "alive",
      "alive",
    ]);
  });

  it("counts misses right away when the startup wait saw the backend come up", async () => {
    // A backend that dies before the first probe still has to be caught.
    expect(
      await probeOutcomes(await closedPort(), 3, { upAtStart: true }),
    ).toEqual(["alive", "alive", "dead"]);
  });
});

describe("docker entrypoint: exit status", () => {
  let workDir: string;

  beforeEach(() => {
    workDir = mkdtempSync(path.join(tmpdir(), "entrypoint-exit-"));
  });

  afterEach(() => {
    rmSync(workDir, { recursive: true, force: true });
  });

  /** The exit-status block, with a log() that records lines to a file. */
  function prelude(serviceStopTimeout = 8): string {
    return [
      "set -uo pipefail",
      "PIDS=()",
      `SERVICE_STOP_TIMEOUT=${serviceStopTimeout}`,
      `log() { printf '%s\\n' "$*" >> "${workDir}/log"; }`,
      exitStatusBlock,
    ].join("\n");
  }

  /**
   * Start a tracked service and wait until it is running its own command, so
   * a signal sent to it cannot land between fork and exec (bash 3.2 loses it).
   */
  function trackedService(body = "exec sleep 30"): string {
    const ready = `${workDir}/service.pid`;
    return [
      `sh -c 'echo $$ > "$1"; ${body}' _ "${ready}" & PIDS+=($!)`,
      `until [ -s "${ready}" ]; do sleep 0.01; done`,
    ].join("\n");
  }

  /**
   * Run `script` in the background, send it SIGTERM once `marker` exists, and
   * report its exit status and roughly how many seconds it took to exit.
   * Afterwards, kill any keep-alive sleep the script recorded in `sleep.pid`.
   */
  async function terminateWhen(
    script: string,
    marker: string,
  ): Promise<{ status: number; seconds: number }> {
    const scriptPath = path.join(workDir, "under-test.sh");
    writeFileSync(scriptPath, script);
    const result = await runBash(
      [
        `bash "${scriptPath}" >/dev/null 2>&1 &`,
        "pid=$!",
        `until [ -e "${marker}" ]; do sleep 0.01; done`,
        "start=$SECONDS",
        "kill -TERM $pid",
        "wait $pid",
        "status=$?",
        `[ -s "${workDir}/sleep.pid" ] && kill "$(cat "${workDir}/sleep.pid")" 2>/dev/null`,
        'echo "$status $((SECONDS - start))"',
      ].join("\n"),
    );
    const [status, seconds] = result.stdout.split(" ").map(Number);
    return { status, seconds };
  }

  it("carries a failure status out through the EXIT trap", async () => {
    // The keep-alive loop exits 1 when the agent-server is gone. Without the
    // status-preserving trap, cleanup()'s own `exit 0` rewrote that into a
    // clean stop and `restart: on-failure` never fired.
    expect((await runBash(`${prelude()}\nexit 1`)).status).toBe(1);
  });

  it("exits 0 on SIGTERM without waiting out the keep-alive sleep", async () => {
    // `docker stop` sends SIGTERM while the keep-alive loop waits on its sleep.
    const marker = `${workDir}/looping`;
    const script = [
      prelude(),
      trackedService(),
      `while :; do sleep 30 & echo $! > "${workDir}/sleep.pid"; touch "${marker}"; wait $!; done`,
    ].join("\n");

    const { status, seconds } = await terminateWhen(script, marker);

    expect(status).toBe(0);
    expect(seconds).toBeLessThan(5);
  }, 15_000);

  it("keeps the failure status when SIGTERM arrives during cleanup", async () => {
    // A service that takes a moment to stop holds cleanup in its wait; a
    // `docker stop` landing then must not turn the crash into a clean exit.
    const script = [
      prelude(),
      trackedService(
        'trap "sleep 1; exit 0" TERM; while :; do sleep 0.1; done',
      ),
      "exit 1",
    ].join("\n");

    expect((await terminateWhen(script, `${workDir}/log`)).status).toBe(1);
  }, 15_000);

  it("force-stops a service that ignores SIGTERM and keeps the failure status", async () => {
    // A hung agent-server may never act on SIGTERM; cleanup must not wait on
    // it forever, or the container stays up after the liveness check fails.
    const script = [
      prelude(1),
      trackedService('trap "" TERM; exec sleep 30'),
      "exit 1",
    ].join("\n");

    const started = Date.now();
    const { status } = await runBash(script);

    expect(status).toBe(1);
    expect(Date.now() - started).toBeLessThan(5_000);
    const servicePid = readFileSync(`${workDir}/service.pid`, "utf-8").trim();
    expect((await runBash(`kill -0 ${servicePid} 2>/dev/null`)).status).toBe(1);
  }, 15_000);

  it("stops the tracked services before exiting", async () => {
    // The failing path has to take the other services down with it, otherwise a
    // restart leaves the old processes behind.
    const script = [prelude(), trackedService(), "exit 1"].join("\n");

    expect((await runBash(script)).status).toBe(1);

    const servicePid = readFileSync(`${workDir}/service.pid`, "utf-8").trim();
    expect((await runBash(`kill -0 ${servicePid} 2>/dev/null`)).status).toBe(1);
  });
});
