// @vitest-environment node
import {
  createServer,
  request as httpRequest,
  type IncomingMessage,
  type Server,
} from "node:http";
import { mkdtempSync } from "node:fs";
import net from "node:net";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import WebSocket, { WebSocketServer } from "ws";

import {
  ConversationEventStream,
  buildConversationEventStreamUrl,
} from "@openhands/typescript-client/clients";
import { server as mswServer } from "#/mocks/node";

import { createMarsTunnelBridge } from "../../scripts/mars-tunnel-bridge.mjs";
import {
  FIRST_MESSAGE_AUTH_MAX_BYTES,
  FIRST_MESSAGE_AUTH_MAX_PENDING,
  RPC_METHODS,
  WEB_ONLY_RPC_METHODS,
  createMarsWebBridge,
  mountMarsWebBridge,
  rewriteAgentServerUrls,
} from "../../scripts/mars-web-bridge.mjs";

// Lets a test take the home folder away; every other call is the real one.
// A fixed set of network addresses for pairing links, and a home folder a
// test can take away.
vi.mock("node:os", async (importOriginal) => {
  const os = await importOriginal<typeof import("node:os")>();
  const homedir = vi.fn(os.homedir);
  const networkInterfaces = () => ({
    lo: [{ family: "IPv4", internal: true, address: "127.0.0.1" }],
    docker0: [{ family: "IPv4", internal: false, address: "172.17.0.1" }],
    tailscale0: [{ family: "IPv4", internal: false, address: "100.101.1.2" }],
    en0: [
      { family: "IPv6", internal: false, address: "fe80::1" },
      { family: "IPv4", internal: false, address: "192.168.1.20" },
    ],
  });
  return {
    ...os,
    default: { ...os, homedir, networkInterfaces },
    homedir,
    networkInterfaces,
  };
});

const SESSION_ID = "sess_a";

// The proxy (httpxy) streams a request body once its upstream socket
// connects, which MSW's interceptor never signals for a request it bypasses.
// Everything here talks to real loopback servers, so take MSW out of the way.
beforeAll(() => mswServer.close());
afterAll(() => mswServer.listen({ onUnhandledRequest: "bypass" }));

function listen(server: Server): Promise<string> {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as net.AddressInfo;
      resolve(`http://127.0.0.1:${port}`);
    });
  });
}

function closeServer(server: Server): void {
  server.closeAllConnections();
  server.close();
}

/**
 * Stands in for the Agent Server behind a tunnel listener: records what it was
 * sent, answers JSON with its own origin in `conversation_url`, and echoes on
 * its WebSocket.
 */
const UPSTREAM_CONVERSATION_PATH = "/mars-upstream/conversation";
const UPSTREAM_ECHO_PATH = "/mars-upstream/echo";

async function startFakeAgentServer() {
  const seen: { http: IncomingMessage[]; ws: IncomingMessage[] } = {
    http: [],
    ws: [],
  };
  let origin = "";
  let largeBody = "{}";
  const state = { delayUpgradeMs: 0 };
  const server = createServer((req, res) => {
    seen.http.push(req);
    if (req.url === UPSTREAM_CONVERSATION_PATH) {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          id: "c1",
          conversation_url: `${origin}/api/conversations/c1`,
          nested: { url: `${origin}/api/conversations/c1/events` },
          unrelated: `${origin}/api/other`,
        }),
      );
      return;
    }
    if (req.url === "/mars-upstream/large") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(largeBody);
      return;
    }
    if (req.url === "/mars-upstream/reset") {
      req.socket.destroy();
      return;
    }
    if (req.url === UPSTREAM_ECHO_PATH && req.method === "POST") {
      const chunks: Buffer[] = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        res.writeHead(200, { "Content-Type": "text/plain" });
        res.end(Buffer.concat(chunks));
      });
      return;
    }
    res.writeHead(404).end();
  });
  const wss = new WebSocketServer({ noServer: true });
  server.on("upgrade", (req, socket, head) => {
    seen.ws.push(req);
    const accept = () =>
      wss.handleUpgrade(req, socket, head, (ws) => {
        ws.on("message", (data, isBinary) =>
          ws.send(`echo:${data.toString()}`, { binary: isBinary }),
        );
      });
    if (state.delayUpgradeMs > 0) setTimeout(accept, state.delayUpgradeMs);
    else accept();
  });
  origin = await listen(server);
  return {
    origin,
    seen,
    setLargeBody: (body: string) => {
      largeBody = body;
    },
    set delayUpgradeMs(ms: number) {
      state.delayUpgradeMs = ms;
    },
    openSockets: () =>
      [...wss.clients].filter((c) => c.readyState === WebSocket.OPEN).length,
    /** Pings every client; resolves once each has ponged back. */
    ping: (data: string) =>
      Promise.all(
        [...wss.clients].map(
          (c) =>
            new Promise<void>((resolve) => {
              c.once("pong", () => resolve());
              c.ping(data);
            }),
        ),
      ),
    close: () => closeServer(server),
  };
}

/**
 * Opens a WebSocket upgrade by hand, for what `ws` refuses to send (a
 * fragment in the path, a frame after the close). Resolves with the status
 * line and the raw socket.
 */
function rawUpgrade(
  origin: string,
  path: string,
): Promise<{ status: string; socket: net.Socket }> {
  const { port } = new URL(origin);
  return new Promise((resolve, reject) => {
    const socket = net.connect(Number(port), "127.0.0.1", () => {
      socket.write(
        `GET ${path} HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n` +
          "Upgrade: websocket\r\nConnection: Upgrade\r\n" +
          "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" +
          "Sec-WebSocket-Version: 13\r\n\r\n",
      );
    });
    socket.once("error", reject);
    socket.once("data", (data) =>
      resolve({ status: data.toString("latin1").split("\r\n")[0], socket }),
    );
  });
}

/** A masked client text frame. */
function clientTextFrame(text: string): Buffer {
  const payload = Buffer.from(text);
  const mask = Buffer.from([1, 2, 3, 4]);
  const masked = payload.map((byte, i) => byte ^ mask[i % 4]);
  let header: Buffer;
  if (payload.length < 126) {
    header = Buffer.from([0x81, 0x80 | payload.length]);
  } else if (payload.length <= 0xffff) {
    header = Buffer.from([0x81, 0x80 | 126, 0, 0]);
    header.writeUInt16BE(payload.length, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81;
    header[1] = 0x80 | 127;
    header.writeBigUInt64BE(BigInt(payload.length), 2);
  }
  return Buffer.concat([header, mask, Buffer.from(masked)]);
}

/** The status an upgrade was refused with, or 101 if it went through. */
function upgradeStatus(url: string, headers: Record<string, string> = {}) {
  return new Promise<number>((resolve) => {
    const ws = new WebSocket(url, { headers });
    ws.on("error", () => {});
    ws.on("unexpected-response", (_req, res) => resolve(res.statusCode ?? 0));
    ws.on("open", () => {
      resolve(101);
      ws.close();
    });
  });
}

/**
 * The tunnel bridge, minus harness-api: openTunnel "connects" a session to the
 * fake agent server's port, and getTunnel reports it the way the registry
 * does (no `host`), until closeTunnel or signOut drops it.
 */
function fakeBridge(upstreamOrigin: string) {
  const localPort = Number(new URL(upstreamOrigin).port);
  const tunnels = new Map<string, { status: string; localPort: number }>();
  return {
    getAuthState: vi.fn(() => ({
      connections: [],
      active: null,
      isPersistent: false,
      canUseOAuth: true,
    })),
    savePat: vi.fn(async () => ({})),
    signOut: vi.fn(async () => {
      tunnels.clear();
    }),
    listAgentConfigs: vi.fn(async () => ({
      configs: [{ id: "cfg_1", name: "Web agent", agent: "openhands" }],
      nextPageToken: null,
    })),
    listConfigSessions: vi.fn(async () => ({
      sessions: [{ session_id: SESSION_ID, status: "SESSION_STATUS_PAUSED" }],
      nextPageToken: null,
    })),
    createSession: vi.fn(async () => ({
      session_id: SESSION_ID,
      status: "SESSION_STATUS_READY",
    })),
    pauseSession: vi.fn(async () => {}),
    openTunnel: vi.fn(async ({ sessionId }: { sessionId: string }) => {
      tunnels.set(sessionId, { status: "connected", localPort });
      return {
        sessionId,
        status: "connected",
        transport: "tunnel",
        host: upstreamOrigin,
        remotePort: 8000,
        localPort,
        error: undefined,
        upstreamFailure: null,
      };
    }),
    closeTunnel: vi.fn(async (sessionId: string) => {
      tunnels.delete(sessionId);
    }),
    destroySession: vi.fn(async () => {}),
    deleteAgentConfig: vi.fn(async () => {}),
    listSandboxSizes: vi.fn(async () => [
      { slug: "mars-2vcpu-4gb", vcpus: 2, memory_mb: 4096 },
    ]),
    getTunnel: vi.fn((sessionId: string) => {
      const tunnel = tunnels.get(sessionId);
      return tunnel && { sessionId, remotePort: 8000, ...tunnel };
    }),
    dispose: vi.fn(async () => {}),
  };
}

async function startWebServer(marsWeb: ReturnType<typeof createMarsWebBridge>) {
  const server = createServer((req, res) => {
    if (marsWeb.handleHttp(req, res)) return;
    res.writeHead(404).end("fallthrough");
  });
  server.on("upgrade", (req, socket, head) => {
    if (marsWeb.handleUpgrade(req, socket, head)) return;
    socket.destroy();
  });
  const origin = await listen(server);
  return { origin, close: () => closeServer(server) };
}

async function rpc(
  origin: string,
  method: string,
  args: unknown[] = [],
  init: RequestInit = {},
) {
  const response = await fetch(`${origin}/mars/rpc/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    body: JSON.stringify({ args }),
    ...init,
  });
  return { status: response.status, body: await response.json() };
}

describe("rewriteAgentServerUrls", () => {
  it("re-points conversation_url at the proxy prefix, whatever host or prefix the upstream used, and leaves other strings alone", () => {
    const base = "http://localhost:8000/mars/sessions/sess_a";
    expect(
      rewriteAgentServerUrls(
        {
          conversation_url:
            "https://ing-1.nyc3.sandbox.ondigitalocean.com/api/conversations/c1",
          items: [
            {
              conversation_url:
                "http://host/runtime/55313/api/conversations/c1",
            },
          ],
          // A URL that merely appears in content must not be touched.
          message: "see http://10.0.0.5:8000/api/conversations/c1/events",
          list: ["http://h/api/conversations", "http://h/api/other"],
          n: 3,
        },
        base,
      ),
    ).toEqual({
      conversation_url: `${base}/api/conversations/c1`,
      items: [{ conversation_url: `${base}/api/conversations/c1` }],
      message: "see http://10.0.0.5:8000/api/conversations/c1/events",
      list: ["http://h/api/conversations", "http://h/api/other"],
      n: 3,
    });
  });
});

describe("RPC_METHODS", () => {
  it("names only bridge methods, and leaves out the desktop's own sign-in", async () => {
    const bridge = createMarsTunnelBridge({ env: {} }) as unknown as Record<
      string,
      unknown
    >;
    for (const method of RPC_METHODS) {
      if (WEB_ONLY_RPC_METHODS.has(method)) continue;
      expect(typeof bridge[method]).toBe("function");
    }
    expect(RPC_METHODS.has("signInWithOAuth")).toBe(false);
    await (bridge.dispose as () => Promise<void>)();
  });
});

describe("createMarsWebBridge", () => {
  let upstream: Awaited<ReturnType<typeof startFakeAgentServer>>;
  let bridge: ReturnType<typeof fakeBridge>;
  let marsWeb: ReturnType<typeof createMarsWebBridge>;
  let web: Awaited<ReturnType<typeof startWebServer>>;

  beforeEach(async () => {
    upstream = await startFakeAgentServer();
    bridge = fakeBridge(upstream.origin);
    marsWeb = createMarsWebBridge({
      bridge: bridge as never,
      env: {},
      log: () => {},
    });
    web = await startWebServer(marsWeb);
  });

  afterEach(async () => {
    await marsWeb.dispose();
    web.close();
    upstream.close();
  });

  it("answers the health probe and ignores non-MARS paths", async () => {
    expect(await (await fetch(`${web.origin}/mars/health`)).json()).toEqual({
      ok: true,
      authRequired: false,
    });
    expect((await fetch(`${web.origin}/elsewhere`)).status).toBe(404);
    expect(await (await fetch(`${web.origin}/elsewhere`)).text()).toBe(
      "fallthrough",
    );
  });

  it("refuses cross-origin requests before touching the bridge", async () => {
    const { status } = await rpc(web.origin, "getAuthState", [], {
      headers: { Origin: "https://evil.example" },
    });
    expect(status).toBe(403);
    expect(bridge.getAuthState).not.toHaveBeenCalled();
  });

  it("exposes the bridge over POST /mars/rpc, but not the desktop's own sign-in", async () => {
    const same = await rpc(web.origin, "getAuthState", [], {
      headers: { Origin: web.origin },
    });
    expect(same.status).toBe(200);
    expect(same.body.result).toMatchObject({ canUseOAuth: true });

    expect((await rpc(web.origin, "signInWithOAuth")).status).toBe(404);
    expect((await rpc(web.origin, "nope")).status).toBe(404);

    // Destructive session management rides the same allowlisted surface.
    expect((await rpc(web.origin, "destroySession", ["sess_1"])).status).toBe(
      200,
    );
    expect(bridge.destroySession).toHaveBeenCalledWith("sess_1");
    expect((await rpc(web.origin, "deleteAgentConfig", ["cfg_1"])).status).toBe(
      200,
    );
    expect(bridge.deleteAgentConfig).toHaveBeenCalledWith("cfg_1");
    expect((await fetch(`${web.origin}/mars/rpc/getAuthState`)).status).toBe(
      405,
    );
  });

  it("exposes the sandbox size catalog over RPC", async () => {
    const { status, body } = await rpc(web.origin, "listSandboxSizes");

    expect(status).toBe(200);
    expect(body.result).toEqual([
      { slug: "mars-2vcpu-4gb", vcpus: 2, memory_mb: 4096 },
    ]);
    expect(bridge.listSandboxSizes).toHaveBeenCalledTimes(1);
  });

  it("surfaces bridge errors with their status", async () => {
    bridge.savePat.mockRejectedValueOnce(
      Object.assign(new Error("Sign in to DigitalOcean first."), {
        status: 401,
      }),
    );
    const { status, body } = await rpc(web.origin, "savePat", [{ token: "x" }]);
    expect(status).toBe(401);
    expect(body.error).toMatchObject({ message: /Sign in/, status: 401 });
  });

  it("openTunnel answers with the proxied host, not the upstream", async () => {
    const { body } = await rpc(web.origin, "openTunnel", [
      { sessionId: SESSION_ID },
    ]);
    expect(bridge.openTunnel).toHaveBeenCalledWith({ sessionId: SESSION_ID });
    expect(body.result).toMatchObject({
      transport: "tunnel",
      host: `${web.origin}/mars/sessions/${SESSION_ID}`,
    });
  });

  it("proxies HTTP to the connected upstream with agent-server URLs rewritten and no credentials of its own", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);

    const response = await fetch(
      `${web.origin}/mars/sessions/${SESSION_ID}${UPSTREAM_CONVERSATION_PATH}`,
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    const seen = upstream.seen.http[0];
    expect(seen.url).toBe(UPSTREAM_CONVERSATION_PATH);
    expect(seen.headers.authorization).toBeUndefined();
    expect(seen.headers.host).toBe(new URL(upstream.origin).host);
    expect(seen.headers["accept-encoding"]).toBe("identity");
    expect(seen.headers["x-forwarded-for"]).toBeUndefined();
    const proxyBase = `${web.origin}/mars/sessions/${SESSION_ID}`;
    expect(body).toEqual({
      id: "c1",
      conversation_url: `${proxyBase}/api/conversations/c1`,
      // Only the URL fields the renderer follows are rewritten; a URL under
      // any other key is content and passes through.
      nested: { url: `${upstream.origin}/api/conversations/c1/events` },
      unrelated: `${upstream.origin}/api/other`,
    });
  });

  it("forwards request bodies and passes non-JSON responses through", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);

    const response = await fetch(
      `${web.origin}/mars/sessions/${SESSION_ID}${UPSTREAM_ECHO_PATH}`,
      {
        method: "POST",
        body: "payload",
      },
    );

    expect(await response.text()).toBe("payload");
  });

  it("answers an upstream failure with a JSON 502", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);

    const response = await fetch(
      `${web.origin}/mars/sessions/${SESSION_ID}/mars-upstream/reset`,
    );

    expect(response.status).toBe(502);
    expect(response.headers.get("content-type")).toBe("application/json");
    expect(await response.json()).toEqual({
      error: { message: expect.stringMatching(/^upstream request failed: /) },
    });
  });

  it("stops proxying a session once the bridge dropped its tunnel on sign-out", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);
    const url = `${web.origin}/mars/sessions/${SESSION_ID}${UPSTREAM_CONVERSATION_PATH}`;
    expect((await fetch(url)).status).toBe(200);

    await rpc(web.origin, "signOut");

    expect((await fetch(url)).status).toBe(409);
  });

  it("answers 409 for a session that is not connected, and after closeTunnel", async () => {
    expect(
      (
        await fetch(
          `${web.origin}/mars/sessions/sess_x${UPSTREAM_CONVERSATION_PATH}`,
        )
      ).status,
    ).toBe(409);

    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);
    await rpc(web.origin, "closeTunnel", [SESSION_ID]);
    expect(
      (
        await fetch(
          `${web.origin}/mars/sessions/${SESSION_ID}${UPSTREAM_CONVERSATION_PATH}`,
        )
      ).status,
    ).toBe(409);
    expect(bridge.closeTunnel).toHaveBeenCalledWith(SESSION_ID);
  });

  it("proxies the WebSocket and relays both ways", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);
    const wsUrl = `${web.origin.replace("http:", "ws:")}/mars/sessions/${SESSION_ID}/sockets/session/c1?after_seq=0`;
    const client = new WebSocket(wsUrl);
    const messages: string[] = [];
    client.on("message", (data) => messages.push(data.toString()));
    await new Promise<void>((resolve, reject) => {
      client.once("open", () => resolve());
      client.once("error", reject);
    });

    client.send("hello");
    await vi.waitFor(() => expect(messages).toEqual(["echo:hello"]));

    const seen = upstream.seen.ws[0];
    expect(seen.url).toBe("/sockets/session/c1?after_seq=0");
    expect(seen.headers.authorization).toBeUndefined();
    client.close();
  });

  it("refuses a WebSocket to a session that is not connected", async () => {
    const client = new WebSocket(
      `${web.origin.replace("http:", "ws:")}/mars/sessions/sess_x/sockets/session/c1`,
    );
    const outcome = await new Promise<string>((resolve) => {
      client.once("unexpected-response", (_req, res) =>
        resolve(`status ${res.statusCode}`),
      );
      client.once("error", (error) => resolve(`error ${error.message}`));
    });
    expect(outcome).toMatch(/409|Unexpected server response: 409/);
  });

  it("answers 400 to a session path that cannot be decoded instead of crashing", async () => {
    expect((await fetch(`${web.origin}/mars/sessions/%E0/`)).status).toBe(400);
    // The server is still alive afterwards.
    expect((await fetch(`${web.origin}/mars/health`)).status).toBe(200);
  });

  it("refuses a request from a foreign Host on a loopback bind (DNS rebinding)", async () => {
    // fetch() will not send a custom Host, so go through node:http.
    const { port } = new URL(web.origin);
    const status = await new Promise<number>((resolve, reject) => {
      httpRequest(
        {
          host: "127.0.0.1",
          port,
          path: "/mars/health",
          headers: { Host: "evil.example:80" },
        },
        (res) => {
          res.resume();
          resolve(res.statusCode ?? 0);
        },
      )
        .on("error", reject)
        .end();
    });
    expect(status).toBe(403);
  });

  it("streams a JSON body past the rewrite limit through untouched instead of truncating it", async () => {
    const big = "x".repeat(9 * 1024 * 1024);
    upstream.setLargeBody(JSON.stringify({ blob: big }));
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);

    const response = await fetch(
      `${web.origin}/mars/sessions/${SESSION_ID}/mars-upstream/large`,
    );
    const text = await response.text();
    expect(response.status).toBe(200);
    expect(text.length).toBe(big.length + '{"blob":""}'.length);
    expect(JSON.parse(text).blob.length).toBe(big.length);
  });

  it("terminates an upstream WebSocket the browser abandoned before the handshake finished", async () => {
    upstream.delayUpgradeMs = 150;
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);

    const ws = new WebSocket(
      `${web.origin.replace("http", "ws")}/mars/sessions/${SESSION_ID}/mars-upstream/socket`,
    );
    ws.on("error", () => {});
    await new Promise((r) => setTimeout(r, 30));
    ws.terminate();
    await new Promise((r) => setTimeout(r, 300));

    expect(upstream.openSockets()).toBe(0);
  });

  it("dispose() closes both ends of proxied WebSockets and disposes the bridge", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);
    const client = new WebSocket(
      `${web.origin.replace("http:", "ws:")}/mars/sessions/${SESSION_ID}/mars-upstream/socket`,
    );
    client.on("error", () => {});
    await new Promise((resolve) => client.once("open", resolve));
    expect(upstream.openSockets()).toBe(1);
    const closed = new Promise((resolve) => client.once("close", resolve));

    await marsWeb.dispose();

    await closed;
    // The fake bridge's dispose() closes no tunnel, so only the proxy's own
    // teardown can have closed the upstream end.
    await vi.waitFor(() => expect(upstream.openSockets()).toBe(0));
    expect(bridge.dispose).toHaveBeenCalledTimes(1);
  });
});

describe("createMarsWebBridge with a session key", () => {
  const KEY = "sk-test-key";
  let upstream: Awaited<ReturnType<typeof startFakeAgentServer>>;
  let bridge: ReturnType<typeof fakeBridge>;
  let marsWeb: ReturnType<typeof createMarsWebBridge>;
  let web: Awaited<ReturnType<typeof startWebServer>>;

  beforeEach(async () => {
    upstream = await startFakeAgentServer();
    bridge = fakeBridge(upstream.origin);
    marsWeb = createMarsWebBridge({
      bridge: bridge as never,
      env: {},
      key: KEY,
      log: () => {},
    });
    web = await startWebServer(marsWeb);
  });

  afterEach(async () => {
    await marsWeb.dispose();
    web.close();
    upstream.close();
  });

  it("requires the key on RPC and proxy routes, hands out a cookie for it, and keeps health open", async () => {
    expect(await (await fetch(`${web.origin}/mars/health`)).json()).toEqual({
      ok: true,
      authRequired: true,
    });
    expect((await rpc(web.origin, "getAuthState")).status).toBe(401);
    expect(
      (
        await rpc(web.origin, "getAuthState", [], {
          headers: { "X-Session-API-Key": "wrong" },
        })
      ).status,
    ).toBe(401);

    const response = await fetch(`${web.origin}/mars/rpc/getAuthState`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-API-Key": KEY },
      body: JSON.stringify({ args: [] }),
    });
    expect(response.status).toBe(200);
    const cookie = response.headers.get("set-cookie") ?? "";
    expect(cookie).toMatch(
      /^mars_web_auth=[0-9a-f]{64}; HttpOnly; SameSite=Strict; Path=\/mars$/,
    );

    // The cookie alone is enough afterwards (what a browser does on its own).
    const cookiePair = cookie.split(";")[0];
    expect(
      (
        await rpc(web.origin, "getAuthState", [], {
          headers: { Cookie: cookiePair },
        })
      ).status,
    ).toBe(200);
  });

  it("keeps a browser's cookie valid across a restart, and not across a key rotation", async () => {
    const first = await fetch(`${web.origin}/mars/rpc/getAuthState`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-API-Key": KEY },
      body: JSON.stringify({ args: [] }),
    });
    const cookiePair = (first.headers.get("set-cookie") ?? "").split(";")[0];
    const statusAfterRestartWith = async (key: string) => {
      const restarted = createMarsWebBridge({
        bridge: fakeBridge(upstream.origin) as never,
        env: {},
        key,
        log: () => {},
      });
      const restartedWeb = await startWebServer(restarted);
      const { status } = await rpc(restartedWeb.origin, "getAuthState", [], {
        headers: { "Content-Type": "application/json", Cookie: cookiePair },
      });
      await restarted.dispose();
      restartedWeb.close();
      return status;
    };

    expect(await statusAfterRestartWith(KEY)).toBe(200);
    expect(await statusAfterRestartWith("rotated-key")).toBe(401);
  });

  it("neither the key header nor the cookie reaches the session host", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
      headers: { "X-Session-API-Key": KEY },
    });
    const response = await fetch(
      `${web.origin}/mars/sessions/${SESSION_ID}/mars-upstream/conversation`,
      { headers: { "X-Session-API-Key": KEY, Cookie: "canvas=1" } },
    );
    expect(response.status).toBe(200);
    const forwarded = upstream.seen.http.at(-1)!.headers;
    expect(forwarded.authorization).toBeUndefined();
    expect(forwarded["x-session-api-key"]).toBeUndefined();
    expect(forwarded.cookie).toBeUndefined();
  });

  it("neither the key header nor the cookie reaches the session host on the WebSocket upgrade", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
      headers: { "X-Session-API-Key": KEY },
    });
    const ws = new WebSocket(
      `${web.origin.replace("http", "ws")}/mars/sessions/${SESSION_ID}/mars-upstream/socket`,
      { headers: { "X-Session-API-Key": KEY, Cookie: "canvas=1" } },
    );
    await new Promise<void>((resolve, reject) => {
      ws.on("open", () => resolve());
      ws.on("error", reject);
    });
    ws.close();

    const forwarded = upstream.seen.ws.at(-1)!.headers;
    expect(forwarded["x-session-api-key"]).toBeUndefined();
    expect(forwarded.cookie).toBeUndefined();
  });

  it("authenticates the WebSocket upgrade by cookie", async () => {
    const first = await fetch(`${web.origin}/mars/rpc/openTunnel`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-API-Key": KEY },
      body: JSON.stringify({ args: [{ sessionId: SESSION_ID }] }),
    });
    const cookiePair = (first.headers.get("set-cookie") ?? "").split(";")[0];
    const wsUrl = `${web.origin.replace("http", "ws")}/mars/sessions/${SESSION_ID}/mars-upstream/socket`;

    const ws = new WebSocket(wsUrl, { headers: { Cookie: cookiePair } });
    await new Promise<void>((resolve, reject) => {
      ws.on("open", () => resolve());
      ws.on("error", reject);
    });
    ws.close();
  });

  it("serves the five calls an agent uses, as docs/DIGITALOCEAN_AGENTS.md shows them", async () => {
    const asAgent = (method: string, args: unknown[]) =>
      rpc(web.origin, method, args, { headers: { "X-Session-API-Key": KEY } });

    const configs = await asAgent("listAgentConfigs", []);
    const sessions = await asAgent("listConfigSessions", ["cfg_1"]);
    const created = await asAgent("createSession", ["cfg_1", "my-task"]);
    const tunnel = await asAgent("openTunnel", [{ sessionId: SESSION_ID }]);
    const paused = await asAgent("pauseSession", [SESSION_ID]);

    expect(
      [configs, sessions, created, tunnel, paused].map((r) => r.status),
    ).toEqual([200, 200, 200, 200, 200]);
    expect(configs.body.result.configs[0]).toMatchObject({
      id: "cfg_1",
      agent: "openhands",
    });
    expect(bridge.listConfigSessions).toHaveBeenCalledWith("cfg_1");
    expect(bridge.createSession).toHaveBeenCalledWith("cfg_1", "my-task");
    expect(tunnel.body.result.host).toBe(
      `${web.origin}/mars/sessions/${SESSION_ID}`,
    );
    expect(bridge.pauseSession).toHaveBeenCalledWith(SESSION_ID);
    expect((await rpc(web.origin, "listAgentConfigs")).status).toBe(401);
  });

  /** Opens a socket with no key or cookie, as an agent's SDK client does. */
  async function openBare() {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
      headers: { "X-Session-API-Key": KEY },
    });
    const ws = new WebSocket(
      `${web.origin.replace("http", "ws")}/mars/sessions/${SESSION_ID}/mars-upstream/socket`,
    );
    const received: string[] = [];
    ws.on("message", (data) => received.push(data.toString()));
    const closed = new Promise<number>((resolve) =>
      ws.on("close", (code) => resolve(code)),
    );
    await new Promise<void>((resolve, reject) => {
      ws.on("open", () => resolve());
      ws.on("error", reject);
    });
    return { ws, received, closed };
  }

  it("lets an agent's socket in by its first message, and dials the session only then", async () => {
    const { ws, received } = await openBare();
    expect(upstream.seen.ws).toHaveLength(0);

    ws.send(JSON.stringify({ type: "auth", session_api_key: KEY }));
    ws.send("hello");

    await vi.waitFor(() => expect(received).toEqual(["echo:hello"]));
    expect(upstream.seen.ws).toHaveLength(1);
    expect(upstream.seen.ws[0].url).toBe("/mars-upstream/socket");
    ws.close();
  });

  it("lets the TypeScript client's own event stream in with the session key", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
      headers: { "X-Session-API-Key": KEY },
    });
    const received: string[] = [];
    const stream = new ConversationEventStream({
      url: buildConversationEventStreamUrl(
        `${web.origin}/mars/sessions/${SESSION_ID}`,
        "c1",
      ),
      sessionApiKey: KEY,
      createWebSocket: (url) =>
        new WebSocket(url) as unknown as globalThis.WebSocket,
      onMessage: (event) => received.push(String(event.data)),
      reconnect: { enabled: false },
    });

    stream.start();
    await vi.waitFor(() => expect(upstream.seen.ws).toHaveLength(1));
    stream.send("ping");

    await vi.waitFor(() => expect(received).toEqual(["echo:ping"]));
    expect(upstream.seen.ws[0].url).toBe("/sockets/events/c1");
    stream.stop();
  });

  it("closes a keyed socket to an unknown, very long session id without crashing", async () => {
    const longId = `sess_${"x".repeat(200)}`;
    const ws = new WebSocket(
      `${web.origin.replace("http", "ws")}/mars/sessions/${longId}/sockets/session/c1`,
    );
    const closed = new Promise<number>((resolve) =>
      ws.on("close", (code) => resolve(code)),
    );
    await new Promise<void>((resolve, reject) => {
      ws.on("open", () => resolve());
      ws.on("error", reject);
    });

    ws.send(JSON.stringify({ type: "auth", session_api_key: KEY }));

    expect(await closed).toBe(1011);
    // The server is still alive afterwards.
    expect((await fetch(`${web.origin}/mars/health`)).status).toBe(200);
  });

  it("closes a socket whose first message is not the key, without touching the session", async () => {
    const { ws, closed } = await openBare();

    ws.send(JSON.stringify({ type: "auth", session_api_key: "wrong" }));

    expect(await closed).toBe(4001);
    expect(upstream.seen.ws).toHaveLength(0);
  });

  it.each<{ what: string; headers: Record<string, string> }>([
    { what: "a wrong key", headers: { "X-Session-API-Key": "wrong" } },
    { what: "a stale cookie", headers: { Cookie: "mars_web_auth=stale" } },
  ])(
    "refuses $what on the socket upgrade at once, without a login window",
    async ({ headers }) => {
      await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
        headers: { "X-Session-API-Key": KEY },
      });
      const url = `${web.origin.replace("http", "ws")}/mars/sessions/${SESSION_ID}/mars-upstream/socket`;

      expect(await upgradeStatus(url, headers)).toBe(401);
      expect(upstream.seen.ws).toHaveLength(0);
    },
  );

  it("drops a socket that sends more than a login's worth before logging in", async () => {
    const { ws, closed } = await openBare();

    ws.send("x".repeat(FIRST_MESSAGE_AUTH_MAX_BYTES + 1));

    expect(await closed).toBe(1006);
    expect(upstream.seen.ws).toHaveLength(0);
    expect((await fetch(`${web.origin}/mars/health`)).status).toBe(200);
  });

  it("refuses more sockets waiting to log in than it allows, and frees a place when one goes", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
      headers: { "X-Session-API-Key": KEY },
    });
    const url = `${web.origin.replace("http", "ws")}/mars/sessions/${SESSION_ID}/mars-upstream/socket`;
    const waiting = await Promise.all(
      Array.from(
        { length: FIRST_MESSAGE_AUTH_MAX_PENDING },
        () =>
          new Promise<WebSocket>((resolve, reject) => {
            const ws = new WebSocket(url);
            ws.on("open", () => resolve(ws));
            ws.on("error", reject);
          }),
      ),
    );

    expect(await upgradeStatus(url)).toBe(503);

    waiting[0].close();
    await vi.waitFor(async () => expect(await upgradeStatus(url)).toBe(101));
    waiting.forEach((ws) => ws.terminate());
  });

  /** A raw socket refused for a wrong key, which never answers the close. */
  async function refusedSocket() {
    const { status, socket } = await rawUpgrade(
      web.origin,
      `/mars/sessions/${SESSION_ID}/mars-upstream/socket`,
    );
    expect(status).toBe("HTTP/1.1 101 Switching Protocols");
    socket.on("error", () => {});
    const refused = new Promise((resolve) => {
      socket.once("data", resolve);
    });
    socket.write(
      clientTextFrame(
        JSON.stringify({ type: "auth", session_api_key: "wrong" }),
      ),
    );
    // The server's close frame.
    await refused;
    return socket;
  }

  it("keeps a refused socket's place until it is gone, however long its close takes", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
      headers: { "X-Session-API-Key": KEY },
    });
    const url = `${web.origin.replace("http", "ws")}/mars/sessions/${SESSION_ID}/mars-upstream/socket`;
    const refused: net.Socket[] = [];
    for (let i = 0; i < FIRST_MESSAGE_AUTH_MAX_PENDING; i += 1) {
      refused.push(await refusedSocket());
    }

    expect(await upgradeStatus(url)).toBe(503);

    refused.forEach((socket) => socket.destroy());
    await vi.waitFor(async () => expect(await upgradeStatus(url)).toBe(101));
  });

  it("drops a refused socket that keeps sending, before it piles up data", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
      headers: { "X-Session-API-Key": KEY },
    });
    const socket = await refusedSocket();
    const closed = new Promise((resolve) => {
      socket.once("close", resolve);
    });

    socket.write(clientTextFrame("x".repeat(FIRST_MESSAGE_AUTH_MAX_BYTES + 1)));

    await closed;
    expect(upstream.seen.ws).toHaveLength(0);
    expect((await fetch(`${web.origin}/mars/health`)).status).toBe(200);
  });

  it("answers 400 to a socket path with a fragment, and stays up", async () => {
    const { status, socket } = await rawUpgrade(
      web.origin,
      `/mars/sessions/${SESSION_ID}/sockets/events/c1#x`,
    );
    socket.destroy();

    expect(status).toBe("HTTP/1.1 400 Bad Request");
    expect((await fetch(`${web.origin}/mars/health`)).status).toBe(200);
  });

  it("passes the session's pings to the agent and the agent's pongs back", async () => {
    const { ws } = await openBare();
    ws.send(JSON.stringify({ type: "auth", session_api_key: KEY }));
    await vi.waitFor(() => expect(upstream.openSockets()).toBe(1));
    const pinged = new Promise<string>((resolve) =>
      ws.once("ping", (data) => resolve(data.toString())),
    );

    // Resolves only once the agent's own pong has reached the session.
    await upstream.ping("are-you-there");

    expect(await pinged).toBe("are-you-there");
    ws.close();
  });
});

describe("createMarsWebBridge first-message login window", () => {
  it("closes a socket that never sends the key", async () => {
    const upstream = await startFakeAgentServer();
    const bridge = fakeBridge(upstream.origin);
    const marsWeb = createMarsWebBridge({
      bridge: bridge as never,
      env: {},
      key: "k",
      authTimeoutMs: 20,
      log: () => {},
    });
    const web = await startWebServer(marsWeb);
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
      headers: { "X-Session-API-Key": "k" },
    });

    const ws = new WebSocket(
      `${web.origin.replace("http", "ws")}/mars/sessions/${SESSION_ID}/x`,
    );
    const code = await new Promise<number>((resolve) =>
      ws.on("close", (closeCode) => resolve(closeCode)),
    );

    expect(code).toBe(4001);
    expect(upstream.seen.ws).toHaveLength(0);
    await marsWeb.dispose();
    web.close();
    upstream.close();
  });

  it("ignores a key that arrives after the window has closed", async () => {
    const upstream = await startFakeAgentServer();
    const marsWeb = createMarsWebBridge({
      bridge: fakeBridge(upstream.origin) as never,
      env: {},
      key: "k",
      authTimeoutMs: 20,
      log: () => {},
    });
    const web = await startWebServer(marsWeb);
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }], {
      headers: { "X-Session-API-Key": "k" },
    });
    const { status, socket } = await rawUpgrade(
      web.origin,
      `/mars/sessions/${SESSION_ID}/x`,
    );
    expect(status).toBe("HTTP/1.1 101 Switching Protocols");

    // The server has sent its close by now; a raw client can still write.
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
    socket.write(
      clientTextFrame(JSON.stringify({ type: "auth", session_api_key: "k" })),
    );
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });

    expect(upstream.seen.ws).toHaveLength(0);
    socket.destroy();
    await marsWeb.dispose();
    web.close();
    upstream.close();
  });
});

describe("mountMarsWebBridge", () => {
  // A mounted bridge reads its state directory; keep it off the developer's.
  const STATE = { MARS_STATE_DIR: mkdtempSync(join(tmpdir(), "mars-mount-")) };

  // Launchers set MARS_WEB=0 on servers that inherit MARS_TOKEN, so an
  // explicit 0 stays quiet even then.
  it.each([
    {},
    { MARS_WEB_KEY: "k" },
    { MARS_WEB: "" },
    { MARS_WEB: "0", MARS_TOKEN: "dop_v1_test" },
  ])("stays off quietly without MARS_WEB=1: %j", (env) => {
    const log = vi.fn();
    const mounted = mountMarsWebBridge({
      env,
      host: "127.0.0.1",
      sessionApiKey: "stack-key",
      log,
      debug: log,
    });
    expect(mounted).toBeNull();
    expect(log).not.toHaveBeenCalled();
  });

  it.each([
    [{ MARS_WEB: "true" }, /MARS_WEB="true" is not "1"/],
    [{ MARS_WEB: "1 " }, /MARS_WEB="1 " is not "1"/],
    [{ MARS_WEB: '"1"' }, /MARS_WEB="\\"1\\"" is not "1"/],
    [
      { MARS_TOKEN: "dop_v1_test" },
      /MARS_TOKEN is set but MARS_WEB is not "1"/,
    ],
  ])("stays off but warns when it looks meant to be on: %j", (env, message) => {
    const warnings: string[] = [];
    const debugs: string[] = [];
    const mounted = mountMarsWebBridge({
      env,
      host: "127.0.0.1",
      sessionApiKey: "stack-key",
      log: (m) => warnings.push(m),
      debug: (m) => debugs.push(m),
    });
    expect(mounted).toBeNull();
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(message);
    expect(debugs).toEqual([]);
  });

  it("mounts with MARS_WEB=1 on a loopback bind, even without a key", async () => {
    const warnings: string[] = [];
    const debugs: string[] = [];
    const mounted = mountMarsWebBridge({
      env: { ...STATE, MARS_WEB: "1" },
      host: "127.0.0.1",
      log: (m) => warnings.push(m),
      debug: (m) => debugs.push(m),
    });
    expect(mounted).not.toBeNull();
    expect(warnings).toEqual([]);
    expect(debugs[0]).toMatch(/local callers only/);
    await mounted!.dispose();
  });

  it("still starts without a home folder, keeping sign-ins off disk", async () => {
    // HOME unset and no passwd entry, as with `docker run --user <uid>`.
    const realHomedir = vi.mocked(homedir).getMockImplementation()!;
    vi.mocked(homedir).mockImplementation(() => {
      throw new Error("uv_os_homedir returned ENOENT");
    });
    try {
      const mounted = mountMarsWebBridge({
        env: { MARS_WEB: "1" },
        host: "127.0.0.1",
        log: () => {},
        debug: () => {},
      });

      expect(mounted).not.toBeNull();
      expect(mounted!.bridge.getAuthState().storage.kind).not.toBe("file");
      await mounted!.dispose();
    } finally {
      vi.mocked(homedir).mockImplementation(realHomedir);
    }
  });

  it("refuses a non-loopback bind without a key, and mounts with one", async () => {
    // A warning, not a debug line: the operator opted in and DigitalOcean
    // will be missing.
    const warnings: string[] = [];
    const debugs: string[] = [];
    expect(
      mountMarsWebBridge({
        env: { MARS_WEB: "1" },
        host: "0.0.0.0",
        log: (m) => warnings.push(m),
        debug: (m) => debugs.push(m),
      }),
    ).toBeNull();
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/not loopback/);
    expect(debugs).toEqual([]);

    const mounted = mountMarsWebBridge({
      env: { ...STATE, MARS_WEB: "1", MARS_WEB_KEY: "k" },
      host: "0.0.0.0",
      log: () => {},
    });
    expect(mounted).not.toBeNull();
    await mounted!.dispose();
  });
});

describe("createMarsWebBridge sign-in storage", () => {
  const KEY = "sk-test-key";
  const TOKEN = `dop_v1_${"e".repeat(64)}`;
  let harness: Server;
  let apiUrl: string;

  // Stands in for harness-api and the account endpoint.
  beforeEach(async () => {
    harness = createServer((req, res) => {
      res.setHeader("Content-Type", "application/json");
      if (req.url?.startsWith("/v2/agents/sessions")) {
        res.end(JSON.stringify({ sessions: [], next_page_token: "" }));
      } else if (req.url === "/v2/account") {
        res.end(
          JSON.stringify({
            account: {
              email: "you@example.com",
              team: { uuid: "team-1", name: "My Team" },
            },
          }),
        );
      } else {
        res.writeHead(404).end("{}");
      }
    });
    apiUrl = await listen(harness);
  });

  afterEach(() => closeServer(harness));

  async function startCanvas(stateDir: string) {
    const marsWeb = createMarsWebBridge({
      env: {
        MARS_API_BASE_URL: apiUrl,
        OH_SECRET_KEY: "host-key",
        OH_CANVAS_SECRET_STORE: "file",
      },
      key: KEY,
      stateDir,
      log: () => {},
    });
    const web = await startWebServer(marsWeb);
    const call = (method: string, args: unknown[] = []) =>
      rpc(web.origin, method, args, {
        headers: {
          "Content-Type": "application/json",
          "X-Session-API-Key": KEY,
        },
      });
    return {
      call,
      stop: async () => {
        await marsWeb.dispose();
        web.close();
      },
    };
  }

  it("keeps the DigitalOcean sign-in, named after its team, across a restart", async () => {
    const stateDir = mkdtempSync(join(tmpdir(), "mars-canvas-"));
    const before = await startCanvas(stateDir);
    const signedIn = await before.call("savePat", [{ token: TOKEN }]);
    await before.stop();

    const after = await startCanvas(stateDir);
    const { body } = await after.call("getAuthState");
    await after.stop();

    expect(signedIn.body.result.active).toMatchObject({ teamName: "My Team" });
    expect(body.result).toMatchObject({
      active: {
        teamName: "My Team",
        email: "you@example.com",
        isExpired: false,
      },
      storage: { kind: "encrypted-file" },
    });
    expect(JSON.stringify(body)).not.toContain(TOKEN);
  });
});

describe("createMarsWebBridge sign-in with DigitalOcean", () => {
  const KEY = "canvas-key";

  function signInBridge() {
    return {
      getAuthState: vi.fn(() => ({ connections: [], active: null })),
      beginSignIn: vi.fn(({ redirectUri }: { redirectUri: string }) => ({
        attemptId: "attempt-1",
        authorizeUrl: `https://cloud.digitalocean.com/v1/oauth/authorize?redirect_uri=${encodeURIComponent(redirectUri)}`,
      })),
      completeSignIn: vi.fn(async () => ({
        active: { teamName: "My Team" },
      })),
      getTunnel: vi.fn(() => undefined),
      dispose: vi.fn(async () => {}),
    };
  }

  async function start(host: string) {
    const bridge = signInBridge();
    const marsWeb = createMarsWebBridge({
      bridge: bridge as never,
      env: {},
      host,
      key: KEY,
      log: () => {},
    });
    const web = await startWebServer(marsWeb);
    return {
      bridge,
      web,
      close: async () => {
        await marsWeb.dispose();
        web.close();
      },
    };
  }

  /** A keyed RPC sent with a chosen Host header, which fetch cannot set. */
  function rpcAs(origin: string, hostHeader: string, method: string) {
    return new Promise<{ status: number; body: { result?: unknown } }>(
      (resolve, reject) => {
        const req = httpRequest(
          `${origin}/mars/rpc/${method}`,
          {
            method: "POST",
            headers: {
              Host: hostHeader,
              "Content-Type": "application/json",
              "X-Session-API-Key": KEY,
            },
          },
          (res) => {
            let text = "";
            res.on("data", (chunk) => {
              text += chunk;
            });
            res.on("end", () =>
              resolve({ status: res.statusCode ?? 0, body: JSON.parse(text) }),
            );
          },
        );
        req.on("error", reject);
        req.end(JSON.stringify({ args: [] }));
      },
    );
  }

  it("sends a browser on this machine back to the page's own origin", async () => {
    const { bridge, web, close } = await start("127.0.0.1");

    const { status, body } = await rpc(web.origin, "beginSignIn", [], {
      headers: { "X-Session-API-Key": KEY },
    });

    expect(status).toBe(200);
    expect(bridge.beginSignIn).toHaveBeenCalledWith({
      redirectUri: `${web.origin}/mars/oauth/callback`,
    });
    expect(body.result).toMatchObject({
      attemptId: "attempt-1",
      returnsHere: true,
    });
    await close();
  });

  it("sends a browser on another device to loopback, and says it will not come back by itself", async () => {
    const { bridge, web, close } = await start("0.0.0.0");
    const port = new URL(web.origin).port;

    const { status, body } = await rpcAs(
      web.origin,
      `192.168.1.5:${port}`,
      "beginSignIn",
    );

    expect(status).toBe(200);
    expect(bridge.beginSignIn).toHaveBeenCalledWith({
      redirectUri: `http://127.0.0.1:${port}/mars/oauth/callback`,
    });
    expect(body.result).toMatchObject({ returnsHere: false });
    await close();
  });

  it("finishes a sign-in at the return address without the key, naming the team only", async () => {
    const { bridge, web, close } = await start("127.0.0.1");

    const response = await fetch(
      `${web.origin}/mars/oauth/callback?code=c1&state=s1`,
    );

    expect(response.status).toBe(200);
    expect(bridge.completeSignIn).toHaveBeenCalledWith(
      "/mars/oauth/callback?code=c1&state=s1",
    );
    const page = await response.text();
    expect(page).toContain("Signed in to DigitalOcean (My Team)");
    expect(response.headers.get("set-cookie")).toBeNull();
    await close();
  });

  it("explains a return that does not belong to an open sign-in", async () => {
    const { bridge, web, close } = await start("127.0.0.1");
    bridge.completeSignIn.mockRejectedValueOnce(
      new Error("This sign-in is no longer open. Start again."),
    );

    const response = await fetch(
      `${web.origin}/mars/oauth/callback?code=c1&state=replayed`,
    );

    expect(response.status).toBe(400);
    expect(await response.text()).toContain("no longer open");
    await close();
  });
});

describe("createMarsWebBridge pairing a new device", () => {
  const KEY = "canvas-key";

  async function start(
    host: string,
    basePath?: string,
    sessionApiKey: string | null = null,
  ) {
    const marsWeb = createMarsWebBridge({
      bridge: {
        getTunnel: () => undefined,
        dispose: async () => {},
      } as never,
      env: {},
      host,
      key: KEY,
      sessionApiKey,
      basePath,
      log: () => {},
    });
    const web = await startWebServer(marsWeb);
    return {
      web,
      close: async () => {
        await marsWeb.dispose();
        web.close();
      },
    };
  }

  /** createPairingLink as a keyed page on another address would call it. */
  function createLinkFrom(origin: string, hostHeader: string) {
    return new Promise<{ links: string[]; expiresAt: string }>(
      (resolve, reject) => {
        const req = httpRequest(
          `${origin}/mars/rpc/createPairingLink`,
          {
            method: "POST",
            headers: {
              Host: hostHeader,
              "Content-Type": "application/json",
              "X-Session-API-Key": KEY,
            },
          },
          (res) => {
            let text = "";
            res.on("data", (chunk) => {
              text += chunk;
            });
            res.on("end", () => resolve(JSON.parse(text).result));
          },
        );
        req.on("error", reject);
        req.end(JSON.stringify({ args: [] }));
      },
    );
  }

  const redeem = (origin: string, secret: string) =>
    fetch(`${origin}/mars/pair`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ secret }),
    });

  it("lets one new device in, once, with the key and the bridge cookie", async () => {
    const { web, close } = await start("0.0.0.0");
    const port = new URL(web.origin).port;
    const { links } = await createLinkFrom(web.origin, `192.168.1.20:${port}`);
    const link = new URL(links[0]);

    const first = await redeem(web.origin, link.hash.slice(1));
    const again = await redeem(web.origin, link.hash.slice(1));

    expect(links).toEqual([`http://192.168.1.20:${port}/pair${link.hash}`]);
    expect(first.status).toBe(200);
    expect(await first.json()).toEqual({ result: { sessionApiKey: KEY } });
    expect(first.headers.get("set-cookie")).toMatch(/^mars_web_auth=/);
    expect(again.status).toBe(400);
    expect((await redeem(web.origin, "guessed")).status).toBe(400);
    await close();
  });

  it("pairs when the bridge's key is the agent-server's own", async () => {
    const { web, close } = await start("0.0.0.0", undefined, KEY);
    const port = new URL(web.origin).port;
    const { links } = await createLinkFrom(web.origin, `192.168.1.20:${port}`);

    const redeemed = await redeem(web.origin, new URL(links[0]).hash.slice(1));

    expect(await redeemed.json()).toEqual({ result: { sessionApiKey: KEY } });
    await close();
  });

  it("refuses to pair when MARS_WEB_KEY is not the agent-server's key", async () => {
    // A paired device would keep the bridge's key as its backend key, which
    // the agent-server refuses.
    const { web, close } = await start(
      "0.0.0.0",
      undefined,
      "agent-server-key",
    );

    const { status, body } = await rpc(web.origin, "createPairingLink", [], {
      headers: { "X-Session-API-Key": KEY },
    });

    expect(status).toBe(400);
    expect(body.error.message).toBe(
      "Pairing is off because this Canvas has a separate MARS_WEB_KEY. Enter the session key on the new device instead.",
    );
    await close();
  });

  it("opens the page under the base path Canvas is mounted at", async () => {
    const { web, close } = await start("0.0.0.0", "/canvas");
    const port = new URL(web.origin).port;

    const { links } = await createLinkFrom(web.origin, `192.168.1.20:${port}`);

    expect(new URL(links[0]).pathname).toBe("/canvas/pair");
    await close();
  });

  it("stops working after five minutes", async () => {
    const { web, close } = await start("0.0.0.0");
    const port = new URL(web.origin).port;
    const { links } = await createLinkFrom(web.origin, `192.168.1.20:${port}`);
    const now = Date.now();
    const clock = vi.spyOn(Date, "now").mockReturnValue(now + 5 * 60_000 + 1);

    const late = await redeem(web.origin, new URL(links[0]).hash.slice(1));

    expect(late.status).toBe(400);
    clock.mockRestore();
    await close();
  });

  it("offers this machine's network addresses to a page on loopback, home network first", async () => {
    const { web, close } = await start("0.0.0.0");
    const port = new URL(web.origin).port;

    const { links } = await createLinkFrom(web.origin, `127.0.0.1:${port}`);

    expect(links.map((link) => new URL(link).host)).toEqual([
      `192.168.1.20:${port}`,
      `100.101.1.2:${port}`,
      `172.17.0.1:${port}`,
    ]);
    await close();
  });

  it("offers no link when Canvas listens only on this computer", async () => {
    const { web, close } = await start("127.0.0.1");

    const { status, body } = await rpc(web.origin, "createPairingLink", [], {
      headers: { "X-Session-API-Key": KEY },
    });

    expect(status).toBe(200);
    expect(body.result.links).toEqual([]);
    await close();
  });
});
