// @vitest-environment node
import { createServer, type IncomingMessage, type Server } from "node:http";
import net from "node:net";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import WebSocket, { WebSocketServer } from "ws";

import {
  createMarsWebBridge,
  rewriteAgentServerUrls,
} from "../../scripts/mars-web-bridge.mjs";

const SESSION_ID = "sess_a";
const TOKEN = "Bearer dop_v1_test";

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
 * Stands in for the Agent Server behind an ingress URL: records what it was
 * sent, answers JSON with its own origin in `conversation_url`, and echoes on
 * its WebSocket (recording pings).
 *
 * Its paths deliberately sit outside the agent-server API surface: this
 * project runs with the MSW mock server from vitest.setup.ts, whose
 * `*\/api/conversations/:id` handler matches any host and would answer both
 * the browser-side request and the proxy's upstream request before either
 * reached a real socket. The URL rewrite under test is about response
 * bodies, not request paths, so nothing is lost.
 */
const UPSTREAM_CONVERSATION_PATH = "/mars-upstream/conversation";
const UPSTREAM_ECHO_PATH = "/mars-upstream/echo";

async function startFakeAgentServer() {
  const seen: { http: IncomingMessage[]; ws: IncomingMessage[] } = {
    http: [],
    ws: [],
  };
  const pings: number[] = [];
  let origin = "";
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
    wss.handleUpgrade(req, socket, head, (ws) => {
      ws.on("ping", () => pings.push(Date.now()));
      ws.on("message", (data, isBinary) =>
        ws.send(`echo:${data.toString()}`, { binary: isBinary }),
      );
    });
  });
  origin = await listen(server);
  return { origin, seen, pings, close: () => closeServer(server) };
}

function fakeBridge(upstreamOrigin: string) {
  return {
    getAuthState: vi.fn(() => ({
      connections: [],
      active: null,
      isPersistent: false,
      canUseOAuth: true,
    })),
    savePat: vi.fn(async () => ({})),
    openTunnel: vi.fn(async ({ sessionId }: { sessionId: string }) => ({
      sessionId,
      status: "connected",
      transport: "ingress",
      host: upstreamOrigin,
      ingressUrlId: "ing-1",
      remotePort: 8000,
      localPort: undefined,
      error: undefined,
      upstreamFailure: null,
    })),
    closeTunnel: vi.fn(async () => {}),
    destroySession: vi.fn(async () => {}),
    deleteAgentConfig: vi.fn(async () => {}),
    getTunnel: vi.fn(() => undefined),
    ingressAuthorizationHeader: vi.fn((url: string) =>
      url.startsWith(upstreamOrigin) ? TOKEN : null,
    ),
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
  it("re-points every agent-server URL at the proxy prefix, whatever host the upstream used", () => {
    const base = "http://localhost:8000/mars/sessions/sess_a";
    expect(
      rewriteAgentServerUrls(
        {
          conversation_url:
            "https://ing-1.nyc3.sandbox.ondigitalocean.com/api/conversations/c1",
          guest: "http://10.0.0.5:8000/api/conversations/c1/events?x=1",
          prefixed: "http://host/runtime/55313/api/conversations/c1",
          list: ["http://h/api/conversations", "http://h/api/other"],
          n: 3,
        },
        base,
      ),
    ).toEqual({
      conversation_url: `${base}/api/conversations/c1`,
      guest: `${base}/api/conversations/c1/events?x=1`,
      prefixed: `${base}/api/conversations/c1`,
      list: [`${base}/api/conversations`, "http://h/api/other"],
      n: 3,
    });
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
      pingIntervalMs: 20,
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
    });
    expect((await fetch(`${web.origin}/api/other`)).status).toBe(404);
    expect(await (await fetch(`${web.origin}/api/other`)).text()).toBe(
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
      transport: "ingress",
      host: `${web.origin}/mars/sessions/${SESSION_ID}`,
    });
  });

  it("proxies HTTP to the connected upstream with the bearer added and agent-server URLs rewritten", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);

    const response = await fetch(
      `${web.origin}/mars/sessions/${SESSION_ID}${UPSTREAM_CONVERSATION_PATH}`,
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    const seen = upstream.seen.http[0];
    expect(seen.url).toBe(UPSTREAM_CONVERSATION_PATH);
    expect(seen.headers.authorization).toBe(TOKEN);
    expect(seen.headers.host).toBe(new URL(upstream.origin).host);
    expect(seen.headers["accept-encoding"]).toBe("identity");
    const proxyBase = `${web.origin}/mars/sessions/${SESSION_ID}`;
    expect(body).toEqual({
      id: "c1",
      conversation_url: `${proxyBase}/api/conversations/c1`,
      nested: { url: `${proxyBase}/api/conversations/c1/events` },
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

  it("proxies the WebSocket with the bearer on the upstream handshake, relays both ways, and pings upstream", async () => {
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
    await vi.waitFor(() => expect(upstream.pings.length).toBeGreaterThan(0), {
      timeout: 1_000,
    });

    const seen = upstream.seen.ws[0];
    expect(seen.url).toBe("/sockets/session/c1?after_seq=0");
    expect(seen.headers.authorization).toBe(TOKEN);
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

  it("dispose() forgets connections and disposes the bridge", async () => {
    await rpc(web.origin, "openTunnel", [{ sessionId: SESSION_ID }]);

    await marsWeb.dispose();

    expect(bridge.dispose).toHaveBeenCalledTimes(1);
    expect(
      (
        await fetch(
          `${web.origin}/mars/sessions/${SESSION_ID}${UPSTREAM_CONVERSATION_PATH}`,
        )
      ).status,
    ).toBe(409);
  });
});
