// @vitest-environment node
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import WebSocket from "ws";
import { server as mswServer } from "#/mocks/node";

import {
  fetchGuestSessionKey,
  readGuestSessionKey,
  startGuestFront,
  startGuestTunnel,
} from "../../scripts/mars-guest-front.mjs";
import { serializeForInlineScript } from "../../scripts/static-server.mjs";
import { standInTunnel, startKeyedGuest, statusOf } from "./keyed-guest";

// Everything here talks to real loopback servers, and the proxy streams a
// request body only once its upstream socket connects, which MSW's
// interceptor never signals for a request it bypasses.
beforeAll(() => mswServer.close());
afterAll(() => mswServer.listen({ onUnhandledFrame: "bypass" }));

const cleanups: (() => unknown)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});

function listen(server: Server): Promise<number> {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () =>
      resolve((server.address() as AddressInfo).port),
    );
  });
}

/** A plain server for the cases static-server never produces. */
async function plainServer(handler: Parameters<typeof createServer>[1]) {
  const server = createServer(handler);
  const port = await listen(server);
  cleanups.push(
    () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  );
  return port;
}

async function keyedGuest(key: string | null) {
  const guest = await startKeyedGuest(key);
  cleanups.push(() => guest.close());
  return guest;
}

async function front(
  upstreamPort: number,
  readKey: () => Promise<string | null>,
) {
  const started = await startGuestFront({
    upstreamPort,
    readKey,
    log: () => {},
  });
  cleanups.push(() => started.stop());
  return `http://127.0.0.1:${started.localPort}`;
}

function openSocket(url: string): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    socket.once("open", () => resolve(socket));
    socket.once("error", reject);
    socket.once("unexpected-response", (_req, res) =>
      reject(new Error(`upgrade answered ${res.statusCode}`)),
    );
  });
}

describe("reading the guest's key", () => {
  it("reads the key a Canvas page carries, as static-server writes it", async () => {
    const key = 'guest-"key"-</script> -end';
    const guest = await keyedGuest(key);

    expect(await fetchGuestSessionKey(guest.port)).toBe(key);
    expect(
      readGuestSessionKey(
        `<script>window.__AGENT_CANVAS_SESSION_API_KEY__=${serializeForInlineScript(key)};</script>`,
      ),
    ).toBe(key);
  });

  it("settles on no key for a page without one", async () => {
    const guest = await keyedGuest(null);

    expect(await fetchGuestSessionKey(guest.port)).toBeNull();
  });

  it.each([
    {
      what: "its server info",
      answer: (res: import("node:http").ServerResponse) => {
        res.writeHead(200, { "content-type": "application/json" });
        res.end('{"title":"OpenHands Agent Server"}');
      },
    },
    {
      what: "a redirect",
      answer: (res: import("node:http").ServerResponse) => {
        res.writeHead(302, { location: "/static/" });
        res.end();
      },
    },
  ])(
    "settles on no key when an older guest's agent-server answers / with $what",
    async ({ answer }) => {
      const port = await plainServer((_req, res) => answer(res));

      expect(await fetchGuestSessionKey(port)).toBeNull();
    },
  );

  it("asks again later when the guest gave no whole answer", async () => {
    const failing = await plainServer((_req, res) => {
      res.writeHead(502);
      res.end();
    });
    const gone = createServer();
    const closed = await listen(gone);
    await new Promise((resolve) => gone.close(resolve));

    await expect(fetchGuestSessionKey(failing)).rejects.toThrow(/502/);
    await expect(fetchGuestSessionKey(closed)).rejects.toThrow();
  });
});

describe("startGuestFront", () => {
  it("adds the guest's key to requests and upgrades, replacing the caller's", async () => {
    const guest = await keyedGuest("guest-key");
    const readKey = vi.fn(() => fetchGuestSessionKey(guest.port));
    const base = await front(guest.port, readKey);

    const status = await statusOf(`${base}/api/settings`, {
      "X-Session-API-Key": "canvas-key",
    });
    const socket = await openSocket(
      `${base.replace("http", "ws")}/sockets/events/c1`,
    );
    cleanups.push(() => socket.terminate());
    const echoed = new Promise((resolve) =>
      socket.once("message", (data) => resolve(String(data))),
    );
    socket.send("hello");

    expect(status).toBe(200);
    expect(await echoed).toBe("echo:hello");
    expect(guest.keysSeen).toEqual(["guest-key", "guest-key"]);
    expect(readKey).toHaveBeenCalledTimes(1);
  });

  it("reads the page once for requests that arrive together", async () => {
    const guest = await keyedGuest("guest-key");
    const readKey = vi.fn(() => fetchGuestSessionKey(guest.port));
    const base = await front(guest.port, readKey);

    const statuses = await Promise.all(
      Array.from({ length: 5 }, () => statusOf(`${base}/api/conversations`)),
    );

    expect(statuses).toEqual([200, 200, 200, 200, 200]);
    expect(readKey).toHaveBeenCalledTimes(1);
  });

  it("reads the page again after the guest refuses the key it had", async () => {
    let key = "first-key";
    const port = await plainServer((req, res) => {
      res.writeHead(req.headers["x-session-api-key"] === key ? 200 : 401);
      res.end();
    });
    const readKey = vi.fn(async () => key);
    const base = await front(port, readKey);

    const before = await statusOf(`${base}/api/settings`);
    key = "second-key"; // the guest restarted with a new key
    const refused = await statusOf(`${base}/api/settings`);
    const after = await statusOf(`${base}/api/settings`);

    expect([before, refused, after]).toEqual([200, 401, 200]);
    expect(readKey).toHaveBeenCalledTimes(2);
  });

  it("does not settle on no key when the page could not be read", async () => {
    const port = await plainServer((req, res) => {
      res.writeHead(
        req.headers["x-session-api-key"] === "guest-key" ? 200 : 401,
      );
      res.end();
    });
    const readKey = vi
      .fn<() => Promise<string | null>>()
      .mockRejectedValueOnce(new Error("still booting"))
      .mockResolvedValue("guest-key");
    const base = await front(port, readKey);

    const first = await statusOf(`${base}/api/settings`);
    const second = await statusOf(`${base}/api/settings`);

    expect([first, second]).toEqual([401, 200]);
    expect(readKey).toHaveBeenCalledTimes(2);
  });

  it("adds nothing for a guest whose page carries no key", async () => {
    const guest = await keyedGuest(null);
    const base = await front(guest.port, () =>
      fetchGuestSessionKey(guest.port),
    );

    expect(await statusOf(`${base}/api/settings`)).toBe(200);
    expect(guest.keysSeen).toEqual([undefined]);
  });
});

describe("startGuestTunnel", () => {
  it("answers with the front's port, the only way in, and stops both", async () => {
    const guest = await keyedGuest("guest-key");
    const raw = standInTunnel(guest.port);

    const tunnel = await startGuestTunnel(
      {
        sessionId: "sess_a",
        remotePort: 8000,
        accessToken: "t",
        log: () => {},
      },
      { startTunnel: raw.startTunnel },
    );
    const status = await statusOf(`http://127.0.0.1:${tunnel.localPort}/api/x`);
    tunnel.stop();

    expect(raw.startTunnel).toHaveBeenCalledWith(
      expect.objectContaining({ localPort: 0, address: "127.0.0.1" }),
    );
    expect(tunnel.localPort).not.toBe(guest.port);
    expect(status).toBe(200);
    expect(raw.stop).toHaveBeenCalled();
    await expect(
      statusOf(`http://127.0.0.1:${tunnel.localPort}/api/x`),
    ).rejects.toThrow();
  });

  it("gives the tunnel back when the port asked for is taken", async () => {
    const taken = await plainServer(() => {});
    const raw = standInTunnel(1);

    await expect(
      startGuestTunnel(
        {
          sessionId: "sess_a",
          remotePort: 8000,
          accessToken: "t",
          localPort: taken,
        },
        { startTunnel: raw.startTunnel },
      ),
    ).rejects.toMatchObject({ code: "EADDRINUSE" });
    expect(raw.stop).toHaveBeenCalled();
  });
});
