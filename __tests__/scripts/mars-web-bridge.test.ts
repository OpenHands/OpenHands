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

import { server as mswServer } from "#/mocks/node";

import { MARS_TUNNEL_IPC } from "../../scripts/mars-tunnel-bridge.mjs";
import {
  RPC_METHODS,
  createMarsWebBridge,
  mountMarsWebBridge,
  rewriteAgentServerUrls,
} from "../../scripts/mars-web-bridge.mjs";

// Lets a test take the home folder away; every other call is the real one.
vi.mock("node:os", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:os")>();
  return { ...actual, homedir: vi.fn(actual.homedir) };
});

const SESSION_ID = "sess_a";

// The proxy (httpxy) streams a request body once its upstream socket
// connects, which MSW's interceptor never signals for a request it bypasses.
// Everything here talks to real loopback servers, so take MSW out of the way.
beforeAll(() => mswServer.close());
afterAll(() => mswServer.listen({ onUnhandledFrame: "bypass" }));

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
    close: () => closeServer(server),
  };
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
  it("is a subset of the desktop IPC surface and leaves OAuth out", () => {
    for (const method of RPC_METHODS) {
      expect(Object.keys(MARS_TUNNEL_IPC)).toContain(method);
    }
    expect(RPC_METHODS.has("signInWithOAuth")).toBe(false);
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

  it("exposes the bridge over POST /mars/rpc and never offers OAuth", async () => {
    const same = await rpc(web.origin, "getAuthState", [], {
      headers: { Origin: web.origin },
    });
    expect(same.status).toBe(200);
    expect(same.body.result).toMatchObject({ canUseOAuth: false });

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

    const refused = new WebSocket(wsUrl);
    const refusedStatus = await new Promise<number>((resolve) => {
      refused.on("unexpected-response", (_r, res) =>
        resolve(res.statusCode ?? 0),
      );
      refused.on("error", () => {});
    });
    expect(refusedStatus).toBe(401);

    const ws = new WebSocket(wsUrl, { headers: { Cookie: cookiePair } });
    await new Promise<void>((resolve, reject) => {
      ws.on("open", () => resolve());
      ws.on("error", reject);
    });
    ws.close();
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
