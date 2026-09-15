// @vitest-environment node
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

import {
  parseTunnelReadyLine,
  resolveMarsTunnelScriptPath,
  startPortForwardTunnel,
} from "../../scripts/tunnel-client.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

describe("resolveMarsTunnelScriptPath", () => {
  it("points at tools/mars_tunnel.py, which exists", () => {
    const scriptPath = resolveMarsTunnelScriptPath();
    expect(scriptPath).toBe(path.join(repoRoot, "tools", "mars_tunnel.py"));
    expect(existsSync(scriptPath)).toBe(true);
  });
});

describe("parseTunnelReadyLine", () => {
  it("parses mars_tunnel.py's readiness line", () => {
    expect(parseTunnelReadyLine("TUNNEL_READY 54321")).toBe(54321);
    expect(parseTunnelReadyLine("TUNNEL_READY 54321\n")).toBe(54321);
  });

  it("returns null for unrelated output", () => {
    expect(parseTunnelReadyLine("[mars-tunnel] forwarding ...")).toBeNull();
    expect(parseTunnelReadyLine("")).toBeNull();
  });
});

/**
 * Build a spawnFn standing in for `uv run tools/mars_tunnel.py`: a small
 * Node script that opens a real TCP listener and announces it exactly like
 * mars_tunnel.py does. Exercises the JS-side stdout-parsing +
 * TCP-health-check + process-lifecycle path without depending on uv/Python
 * being installed. The real Python implementation is covered separately by
 * the (uv-gated) "real mars_tunnel.py" suite below.
 */
function fakeTunnelSpawnFn() {
  const script = `
    const net = require("node:net");
    const server = net.createServer((socket) => socket.end());
    server.listen(0, "127.0.0.1", () => {
      console.log(\`TUNNEL_READY \${server.address().port}\`);
    });
    process.on("SIGTERM", () => process.exit(0));
  `;
  return (_command, _args, options) =>
    spawn(process.execPath, ["-e", script], options);
}

function fakeFailingTunnelSpawnFn(stderrMessage) {
  const script = `
    console.error(${JSON.stringify(stderrMessage)});
    process.exit(1);
  `;
  return (_command, _args, options) =>
    spawn(process.execPath, ["-e", script], options);
}

describe("startPortForwardTunnel (JS orchestration, fake subprocess)", () => {
  it("resolves once the subprocess announces a port and the listener is healthy", async () => {
    const tunnel = await startPortForwardTunnel({
      sessionId: "sess_abc123",
      remotePort: 8000,
      accessToken: "test-token",
      spawnFn: fakeTunnelSpawnFn(),
    });

    try {
      expect(tunnel.sessionId).toBe("sess_abc123");
      expect(tunnel.remotePort).toBe(8000);
      expect(tunnel.localPort).toBeGreaterThan(0);

      await new Promise<void>((resolve, reject) => {
        const socket = net.connect(
          { port: tunnel.localPort, host: "127.0.0.1" },
          () => {
            socket.end();
            resolve();
          },
        );
        socket.once("error", reject);
      });
    } finally {
      tunnel.stop();
    }
  });

  it("passes the access token via env, not argv", async () => {
    const script = `
      const net = require("node:net");
      if (process.env.MARS_TUNNEL_ACCESS_TOKEN !== "secret-token") {
        console.error("missing or wrong token");
        process.exit(1);
      }
      const server = net.createServer((socket) => socket.end());
      server.listen(0, "127.0.0.1", () => {
        console.log(\`TUNNEL_READY \${server.address().port}\`);
      });
      process.on("SIGTERM", () => process.exit(0));
    `;
    const tunnel = await startPortForwardTunnel({
      sessionId: "sess_abc123",
      remotePort: 8000,
      accessToken: "secret-token",
      spawnFn: (_command, args, options) => {
        expect(args.join(" ")).not.toContain("secret-token");
        return spawn(process.execPath, ["-e", script], options);
      },
    });
    tunnel.stop();
  });

  it("rejects when sessionId is missing", async () => {
    await expect(
      startPortForwardTunnel({ remotePort: 8000, accessToken: "test-token" }),
    ).rejects.toThrow(/sessionId/);
  });

  it("rejects when accessToken is missing", async () => {
    await expect(
      startPortForwardTunnel({ sessionId: "sess_abc123", remotePort: 8000 }),
    ).rejects.toThrow(/accessToken/);
  });

  it("rejects with the subprocess's stderr tail when it exits before announcing a port", async () => {
    await expect(
      startPortForwardTunnel({
        sessionId: "sess_abc123",
        remotePort: 8000,
        accessToken: "test-token",
        spawnFn: fakeFailingTunnelSpawnFn(
          "server rejected tunnel (403 Forbidden): invalid token",
        ),
      }),
    ).rejects.toThrow(/invalid token/);
  });
});

function hasUv() {
  const result = spawnSync("uv", ["--version"], { stdio: "ignore" });
  return !result.error && result.status === 0;
}

/**
 * End-to-end coverage of the real tools/mars_tunnel.py, run via the actual
 * `uv run` the Electron main process uses — against a local fake harness-api
 * standing in for MARS's port-forward WebSocket. Skipped when `uv` isn't on
 * PATH (CI's unit-test job doesn't install it; only its separate e2e job
 * does) — this suite is for local verification, not a CI gate.
 */
describe.skipIf(!hasUv())("startPortForwardTunnel (real mars_tunnel.py via uv run)", () => {
  const sessionId = "sess_abc123";
  const remotePort = 8000;
  const expectedPath = `/v2/agents/sessions/${sessionId}/port-forward/${remotePort}`;

  function startFakeHarness({ expectToken }) {
    const script = `
# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets>=13,<18"]
# ///
import asyncio
from websockets.asyncio.server import serve
from websockets.http11 import Response
from websockets.datastructures import Headers

def process_request(connection, request):
    if request.path != ${JSON.stringify(expectedPath)}:
        return connection.respond(404, "not found\\n")
    auth = request.headers.get("Authorization")
    if auth != "Bearer ${expectToken}":
        return Response(403, "Forbidden", Headers(), b'{"message":"invalid token"}')
    return None

async def handler(ws):
    async for message in ws:
        payload = message if isinstance(message, (bytes, bytearray)) else message.encode()
        await ws.send(b"echo:" + payload)

async def main():
    async with serve(handler, "127.0.0.1", 0, process_request=process_request) as server:
        port = server.sockets[0].getsockname()[1]
        print(f"HARNESS_READY {port}", flush=True)
        await asyncio.Future()

asyncio.run(main())
`;
    const proc = spawn("uv", ["run", "-"], { stdio: ["pipe", "pipe", "pipe"] });
    proc.stdin.write(script);
    proc.stdin.end();
    return proc;
  }

  function waitForLine(child, regex, timeoutMs = 20_000) {
    return new Promise((resolve, reject) => {
      let buffer = "";
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error(`Timed out waiting for ${regex} in: ${buffer}`));
      }, timeoutMs);
      function onData(chunk) {
        buffer += chunk.toString();
        const match = regex.exec(buffer);
        if (match) {
          cleanup();
          resolve(match);
        }
      }
      function onError(err) {
        cleanup();
        reject(err);
      }
      function cleanup() {
        clearTimeout(timer);
        child.stdout.off("data", onData);
        child.off("error", onError);
      }
      child.stdout.on("data", onData);
      child.once("error", onError);
    });
  }

  let harness;
  let tunnel;

  afterEach(() => {
    tunnel?.stop();
    tunnel = undefined;
    harness?.kill("SIGTERM");
    harness = undefined;
  });

  it(
    "bridges a local TCP connection through to the fake guest and back",
    async () => {
      harness = startFakeHarness({ expectToken: "test-token" });
      const [, harnessPort] = await waitForLine(harness, /HARNESS_READY (\d+)/);

      tunnel = await startPortForwardTunnel({
        sessionId,
        remotePort,
        accessToken: "test-token",
        apiUrl: `http://127.0.0.1:${harnessPort}`,
      });

      expect(tunnel.localPort).toBeGreaterThan(0);

      const response = await new Promise<Buffer>((resolve, reject) => {
        const socket = net.connect(
          { port: tunnel.localPort, host: "127.0.0.1" },
          () => socket.write("hello-mars"),
        );
        socket.once("data", (data) => {
          socket.end();
          resolve(data);
        });
        socket.once("error", reject);
      });

      expect(response.toString()).toBe("echo:hello-mars");
    },
    30_000,
  );

  it(
    "surfaces the server's rejection message for a bad token",
    async () => {
      harness = startFakeHarness({ expectToken: "test-token" });
      const [, harnessPort] = await waitForLine(harness, /HARNESS_READY (\d+)/);

      tunnel = await startPortForwardTunnel({
        sessionId,
        remotePort,
        accessToken: "wrong-token",
        apiUrl: `http://127.0.0.1:${harnessPort}`,
      });

      const stderrChunks: Buffer[] = [];
      tunnel.process.stderr.on("data", (chunk) => stderrChunks.push(chunk));

      await new Promise<void>((resolve, reject) => {
        const socket = net.connect(
          { port: tunnel.localPort, host: "127.0.0.1" },
          () => socket.write("x"),
        );
        socket.once("close", () => resolve());
        socket.once("error", () => resolve());
        setTimeout(() => reject(new Error("timed out waiting for rejection")), 10_000);
      });

      await new Promise((r) => setTimeout(r, 500));
      expect(Buffer.concat(stderrChunks).toString()).toMatch(
        /server rejected tunnel \(403.*invalid token/,
      );
    },
    30_000,
  );
});
