// @vitest-environment node
import { createServer } from "node:http";
import net from "node:net";
import { WebSocketServer } from "ws";
import { afterEach, describe, expect, it } from "vitest";

import { startPortForwardTunnel } from "../../scripts/tunnel-client.mjs";

const sessionId = "sess_abc123";
const remotePort = 8000;
const expectedPath = `/v2/agents/sessions/${sessionId}/port-forward/${remotePort}`;

/**
 * A local stand-in for MARS's port-forward endpoint: rejects the wrong path
 * or a bad bearer token at the HTTP-upgrade level (matching harness-api's
 * real behavior), otherwise upgrades and echoes back whatever it receives —
 * enough to prove bytes really flow both ways through the tunnel.
 */
function startFakeHarness({ expectToken = "test-token" } = {}) {
  const httpServer = createServer((_req, res) => {
    res.writeHead(404).end("not found\n");
  });
  const wss = new WebSocketServer({ noServer: true });

  httpServer.on("upgrade", (req, socket, head) => {
    if (req.url !== expectedPath) {
      socket.write("HTTP/1.1 404 Not Found\r\n\r\n");
      socket.destroy();
      return;
    }
    if (req.headers.authorization !== `Bearer ${expectToken}`) {
      const body = JSON.stringify({ message: "invalid token" });
      socket.write(
        `HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`,
      );
      socket.destroy();
      return;
    }
    wss.handleUpgrade(req, socket, head, (ws) => {
      ws.on("message", (data) =>
        ws.send(Buffer.concat([Buffer.from("echo:"), data as Buffer])),
      );
    });
  });

  return new Promise<{ port: number; close: () => void }>((resolve) => {
    httpServer.listen(0, "127.0.0.1", () => {
      const { port } = httpServer.address() as net.AddressInfo;
      resolve({
        port,
        close: () => {
          wss.close();
          httpServer.close();
        },
      });
    });
  });
}

describe("startPortForwardTunnel", () => {
  let harness: { port: number; close: () => void } | undefined;
  let tunnel: Awaited<ReturnType<typeof startPortForwardTunnel>> | undefined;

  afterEach(() => {
    tunnel?.stop();
    tunnel = undefined;
    harness?.close();
    harness = undefined;
  });

  it("rejects when sessionId is missing", async () => {
    await expect(
      // @ts-expect-error deliberately omitting a required field to test runtime validation
      startPortForwardTunnel({ remotePort, accessToken: "test-token" }),
    ).rejects.toThrow(/sessionId/);
  });

  it("rejects when accessToken is missing", async () => {
    await expect(
      // @ts-expect-error deliberately omitting a required field to test runtime validation
      startPortForwardTunnel({ sessionId, remotePort }),
    ).rejects.toThrow(/accessToken/);
  });

  it("rejects an out-of-range remote port", async () => {
    await expect(
      startPortForwardTunnel({
        sessionId,
        remotePort: 70000,
        accessToken: "test-token",
      }),
    ).rejects.toThrow(/Invalid remote port/);
  });

  it("bridges a local TCP connection through to the guest and back", async () => {
    harness = await startFakeHarness({ expectToken: "test-token" });

    tunnel = await startPortForwardTunnel({
      sessionId,
      remotePort,
      accessToken: "test-token",
      apiUrl: `http://127.0.0.1:${harness.port}`,
    });

    expect(tunnel.sessionId).toBe(sessionId);
    expect(tunnel.remotePort).toBe(remotePort);
    expect(tunnel.localPort).toBeGreaterThan(0);

    const response = await new Promise<Buffer>((resolve, reject) => {
      const socket = net.connect(
        { port: tunnel!.localPort, host: "127.0.0.1" },
        () => socket.write("hello-mars"),
      );
      socket.once("data", (data: Buffer) => {
        socket.end();
        resolve(data);
      });
      socket.once("error", reject);
    });

    expect(response.toString()).toBe("echo:hello-mars");
  });

  it("relays a large payload intact under backpressure in both directions", async () => {
    // A raw byte-for-byte echo (no "echo:" prefix) so the received buffer can
    // be compared directly against what was sent.
    const wss = new WebSocketServer({ noServer: true });
    const rawHarness = createServer((_req, res) => res.writeHead(404).end());
    rawHarness.on("upgrade", (req, socket, head) => {
      wss.handleUpgrade(req, socket, head, (ws) => {
        ws.on("message", (data) => ws.send(data as Buffer));
      });
    });
    const rawPort = await new Promise<number>((resolve) => {
      rawHarness.listen(0, "127.0.0.1", () =>
        resolve((rawHarness.address() as net.AddressInfo).port),
      );
    });
    harness = { port: rawPort, close: () => { wss.close(); rawHarness.close(); } };

    tunnel = await startPortForwardTunnel({
      sessionId,
      remotePort,
      accessToken: "test-token",
      apiUrl: `http://127.0.0.1:${harness.port}`,
    });

    // Several MB, well beyond a single TCP read chunk, so the tunnel's
    // localSocket "data" handler fires many times — if pause()/resume()
    // ever got mismatched (paused but never resumed), this would hang and
    // the test would time out rather than silently pass.
    const payload = Buffer.alloc(4 * 1024 * 1024);
    for (let i = 0; i < payload.length; i += 1) payload[i] = i % 256;

    const received = await new Promise<Buffer>((resolve, reject) => {
      const socket = net.connect({ port: tunnel!.localPort, host: "127.0.0.1" });
      const chunks: Buffer[] = [];
      let total = 0;
      socket.on("connect", () => socket.end(payload));
      socket.on("data", (chunk: Buffer) => {
        chunks.push(chunk);
        total += chunk.length;
        if (total >= payload.length) {
          socket.end();
          resolve(Buffer.concat(chunks));
        }
      });
      socket.once("error", reject);
    });

    expect(received.equals(payload)).toBe(true);
  }, 15_000);

  it("surfaces the server's rejection message for a bad token", async () => {
    harness = await startFakeHarness({ expectToken: "test-token" });

    const logs: string[] = [];
    tunnel = await startPortForwardTunnel({
      sessionId,
      remotePort,
      accessToken: "wrong-token",
      apiUrl: `http://127.0.0.1:${harness.port}`,
      log: (message) => logs.push(message),
    });

    await new Promise<void>((resolve, reject) => {
      const socket = net.connect(
        { port: tunnel!.localPort, host: "127.0.0.1" },
        () => socket.write("x"),
      );
      socket.once("close", () => resolve());
      socket.once("error", () => resolve());
      setTimeout(() => reject(new Error("timed out waiting for rejection")), 5_000);
    });

    expect(logs.join("\n")).toMatch(/server rejected tunnel \(403.*invalid token/);
  });

  it("closes active connections and stops accepting new ones on stop()", async () => {
    harness = await startFakeHarness({ expectToken: "test-token" });

    tunnel = await startPortForwardTunnel({
      sessionId,
      remotePort,
      accessToken: "test-token",
      apiUrl: `http://127.0.0.1:${harness.port}`,
    });
    const { localPort } = tunnel;

    const socket = net.connect({ port: localPort, host: "127.0.0.1" });
    // stop() destroys the connection abruptly (RST) rather than a graceful
    // FIN, so the client side sees ECONNRESET, not a clean end.
    socket.on("error", () => {});
    await new Promise((resolve) => socket.once("connect", resolve));

    const closed = new Promise((resolve) => socket.once("close", resolve));
    tunnel.stop();
    await closed;

    await expect(
      new Promise((resolve, reject) => {
        const retry = net.connect({ port: localPort, host: "127.0.0.1" });
        retry.once("connect", () => {
          retry.end();
          resolve(undefined);
        });
        retry.once("error", reject);
      }),
    ).rejects.toThrow();
  });
});
