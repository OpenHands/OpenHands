// @vitest-environment node
import { createServer } from "node:http";
import net from "node:net";
import { WebSocketServer } from "ws";
import { describe, expect, it, vi } from "vitest";

import { createTunnelRegistry } from "../../scripts/tunnel-registry.mjs";
import { startPortForwardTunnel } from "../../scripts/tunnel-client.mjs";

/** A stub ensureAwake that resolves immediately, for tests only exercising tunnel logic. */
const noopEnsureAwake = async () => ({ status: "SESSION_STATUS_READY" });

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (err: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** A stub startTunnel that hands out an incrementing fake port per call. */
function fakeStartTunnel() {
  let nextPort = 40000;
  const calls: unknown[] = [];
  const fn = vi.fn(async (options) => {
    calls.push(options);
    return {
      sessionId: options.sessionId,
      remotePort: options.remotePort,
      localPort: nextPort++,
      getLastUpstreamFailure: () => null,
      stop: vi.fn(),
    };
  });
  return { fn, calls };
}

describe("createTunnelRegistry", () => {
  it("attaches a session and reports it connected", async () => {
    const { fn } = fakeStartTunnel();
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });

    const result = await registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "token",
    });

    expect(result).toMatchObject({ sessionId: "sess_a", status: "connected" });
    expect(result.localPort).toBeGreaterThan(0);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("reuses an existing tunnel instead of opening a duplicate", async () => {
    const { fn } = fakeStartTunnel();
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });

    const first = await registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
    });
    const second = await registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
    });

    expect(second.localPort).toBe(first.localPort);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("dedupes concurrent attach() calls for the same session into one tunnel", async () => {
    const gate = deferred<void>();
    const fn = vi.fn(async (options) => {
      await gate.promise;
      return {
        sessionId: options.sessionId,
        remotePort: options.remotePort,
        localPort: 41000,
        getLastUpstreamFailure: () => null,
        stop: vi.fn(),
      };
    });
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });

    const attempt1 = registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
    });
    const attempt2 = registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
    });
    gate.resolve();

    const [r1, r2] = await Promise.all([attempt1, attempt2]);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(r1.localPort).toBe(r2.localPort);
  });

  it("gives independent sessions independent, non-colliding ports", async () => {
    const { fn } = fakeStartTunnel();
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });

    const [a, b] = await Promise.all([
      registry.attach({
        sessionId: "sess_a",
        remotePort: 8000,
        getAccessToken: () => "t",
      }),
      registry.attach({
        sessionId: "sess_b",
        remotePort: 8000,
        getAccessToken: () => "t",
      }),
    ]);

    expect(a.localPort).not.toBe(b.localPort);
    expect(fn).toHaveBeenCalledTimes(2);
    expect(
      registry
        .list()
        .map((e) => e.sessionId)
        .sort(),
    ).toEqual(["sess_a", "sess_b"]);
  });

  it("detach() tears down only the targeted session", async () => {
    const { fn } = fakeStartTunnel();
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });

    await registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
    });
    await registry.attach({
      sessionId: "sess_b",
      remotePort: 8000,
      getAccessToken: () => "t",
    });

    await registry.detach("sess_a");

    expect(registry.get("sess_a")).toBeUndefined();
    expect(registry.get("sess_b")).toMatchObject({ status: "connected" });
    const aTunnel = await fn.mock.results[0].value;
    expect(aTunnel.stop).toHaveBeenCalledTimes(1);
  });

  it("detachAll() tears down every session", async () => {
    const { fn } = fakeStartTunnel();
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });

    await registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
    });
    await registry.attach({
      sessionId: "sess_b",
      remotePort: 8000,
      getAccessToken: () => "t",
    });

    await registry.detachAll();

    expect(registry.list()).toEqual([]);
    for (const result of fn.mock.results) {
      const tunnel = await result.value;
      expect(tunnel.stop).toHaveBeenCalledTimes(1);
    }
  });

  it("surfaces a failed attach() without affecting other sessions, and allows retry", async () => {
    let badAttempts = 0;
    const fn = vi.fn(async (options) => {
      if (options.sessionId === "sess_bad") {
        badAttempts += 1;
        if (badAttempts === 1) {
          throw new Error(
            "server rejected tunnel (403 Forbidden): invalid token",
          );
        }
      }
      return {
        sessionId: options.sessionId,
        remotePort: options.remotePort,
        localPort: 42000,
        getLastUpstreamFailure: () => null,
        stop: vi.fn(),
      };
    });
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });

    await registry.attach({
      sessionId: "sess_ok",
      remotePort: 8000,
      getAccessToken: () => "t",
    });
    await expect(
      registry.attach({
        sessionId: "sess_bad",
        remotePort: 8000,
        getAccessToken: () => "wrong",
      }),
    ).rejects.toThrow(/invalid token/);

    expect(registry.get("sess_ok")).toMatchObject({ status: "connected" });
    expect(registry.get("sess_bad")).toMatchObject({
      status: "error",
      error: expect.stringMatching(/invalid token/),
    });

    // Retrying (e.g. after the user fixes their token) opens a fresh attempt.
    const retried = await registry.attach({
      sessionId: "sess_bad",
      remotePort: 8000,
      getAccessToken: () => "t",
    });
    expect(retried.status).toBe("connected");
  });

  it("wakes the session again when reusing a connected tunnel", async () => {
    const ensureAwake = vi.fn(noopEnsureAwake);
    const { fn } = fakeStartTunnel();
    const registry = createTunnelRegistry({ startTunnel: fn, ensureAwake });
    const params = {
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
    };

    await registry.attach(params);
    await registry.attach(params);

    expect(ensureAwake).toHaveBeenCalledTimes(2);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("falls back to an OS-picked port when the requested one is taken", async () => {
    const { fn } = fakeStartTunnel();
    fn.mockImplementationOnce(async () => {
      throw Object.assign(new Error("listen EADDRINUSE"), {
        code: "EADDRINUSE",
      });
    });
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });

    const result = await registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
      localPort: 51000,
    });

    expect(fn).toHaveBeenLastCalledWith(
      expect.objectContaining({ localPort: 0 }),
    );
    expect(result).toMatchObject({ status: "connected", localPort: 40000 });
  });

  it("detachOwnedBy() tears down only that owner's tunnels", async () => {
    const { fn } = fakeStartTunnel();
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });
    await registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
      owner: "conn_a",
    });
    await registry.attach({
      sessionId: "sess_b",
      remotePort: 8000,
      getAccessToken: () => "t",
      owner: "conn_b",
    });

    await registry.detachOwnedBy("conn_a");

    expect(registry.list().map((e) => e.sessionId)).toEqual(["sess_b"]);
  });

  it("rejects when sessionId is missing", async () => {
    const { fn } = fakeStartTunnel();
    const registry = createTunnelRegistry({
      startTunnel: fn,
      ensureAwake: noopEnsureAwake,
    });
    await expect(
      // @ts-expect-error deliberately omitting a required field to test runtime validation
      registry.attach({ remotePort: 8000, getAccessToken: () => "t" }),
    ).rejects.toThrow(/sessionId/);
  });

  it("waits for the session to be awake before dialing the tunnel", async () => {
    const order: string[] = [];
    const ensureAwake = vi.fn(async () => {
      order.push("ensureAwake");
      return { status: "SESSION_STATUS_READY" };
    });
    const { fn } = fakeStartTunnel();
    fn.mockImplementation(async (options) => {
      order.push("startTunnel");
      return {
        sessionId: options.sessionId,
        remotePort: options.remotePort,
        localPort: 43000,
        getLastUpstreamFailure: () => null,
        stop: vi.fn(),
      };
    });
    const registry = createTunnelRegistry({ startTunnel: fn, ensureAwake });

    await registry.attach({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: () => "t",
    });

    expect(order).toEqual(["ensureAwake", "startTunnel"]);
    expect(ensureAwake).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId: "sess_a" }),
    );
  });

  it("surfaces a failure to wake the session without ever dialing the tunnel", async () => {
    const ensureAwake = vi.fn(async () => {
      throw new Error(
        "Session sess_a is SESSION_STATUS_FAILED and cannot be connected to.",
      );
    });
    const { fn } = fakeStartTunnel();
    const registry = createTunnelRegistry({ startTunnel: fn, ensureAwake });

    await expect(
      registry.attach({
        sessionId: "sess_a",
        remotePort: 8000,
        getAccessToken: () => "t",
      }),
    ).rejects.toThrow(/SESSION_STATUS_FAILED/);

    expect(fn).not.toHaveBeenCalled();
    expect(registry.get("sess_a")).toMatchObject({
      status: "error",
      error: expect.stringMatching(/SESSION_STATUS_FAILED/),
    });
  });
});

/**
 * End-to-end: the real startPortForwardTunnel, against two independent fake
 * harness-api servers standing in for two concurrently attached MARS
 * sessions, proving the registry's dedup/port-allocation logic holds up
 * against the real tunnel client, not just a stub.
 */
describe("createTunnelRegistry (real tunnel client)", () => {
  function startFakeHarness(
    sessionId: string,
    remotePort: number,
    expectToken: string,
  ) {
    const expectedPath = `/v2/agents/sessions/${sessionId}/port-forward/${remotePort}`;
    const sessionPath = `/v2/agents/sessions/${sessionId}`;
    const httpServer = createServer((req, res) => {
      // Also serves the plain REST session-status endpoint ensureSessionAwake
      // hits before the tunnel is dialed — always READY here, so this test
      // stays focused on the tunnel/registry path rather than resume logic
      // (covered separately in mars-session.test.ts).
      if (req.method === "GET" && req.url === sessionPath) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            session: { session_id: sessionId, status: "SESSION_STATUS_READY" },
          }),
        );
        return;
      }
      res.writeHead(404).end();
    });
    const wss = new WebSocketServer({ noServer: true });

    httpServer.on("upgrade", (req, socket, head) => {
      if (
        req.url !== expectedPath ||
        req.headers.authorization !== `Bearer ${expectToken}`
      ) {
        socket.write("HTTP/1.1 403 Forbidden\r\n\r\n");
        socket.destroy();
        return;
      }
      wss.handleUpgrade(req, socket, head, (ws) => {
        ws.on("message", (data) =>
          ws.send(
            Buffer.concat([Buffer.from(`echo:${sessionId}:`), data as Buffer]),
          ),
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

  it("bridges two concurrently attached sessions to their own independent guests", async () => {
    const harnessA = await startFakeHarness("sess_a", 8000, "token-a");
    const harnessB = await startFakeHarness("sess_b", 8000, "token-b");
    const registry = createTunnelRegistry({
      startTunnel: startPortForwardTunnel,
    });

    try {
      const [a, b] = await Promise.all([
        registry.attach({
          sessionId: "sess_a",
          remotePort: 8000,
          getAccessToken: () => "token-a",
          apiUrl: `http://127.0.0.1:${harnessA.port}`,
        }),
        registry.attach({
          sessionId: "sess_b",
          remotePort: 8000,
          getAccessToken: () => "token-b",
          apiUrl: `http://127.0.0.1:${harnessB.port}`,
        }),
      ]);

      expect(a.localPort).not.toBe(b.localPort);

      const send = (port: number, payload: string) =>
        new Promise<Buffer>((resolve, reject) => {
          const socket = net.connect({ port, host: "127.0.0.1" }, () =>
            socket.write(payload),
          );
          socket.once("data", (data: Buffer) => {
            socket.end();
            resolve(data);
          });
          socket.once("error", reject);
        });

      const [respA, respB] = await Promise.all([
        send(a.localPort!, "hi-a"),
        send(b.localPort!, "hi-b"),
      ]);

      expect(respA.toString()).toBe("echo:sess_a:hi-a");
      expect(respB.toString()).toBe("echo:sess_b:hi-b");
    } finally {
      await registry.detachAll();
      harnessA.close();
      harnessB.close();
    }
  });
});
