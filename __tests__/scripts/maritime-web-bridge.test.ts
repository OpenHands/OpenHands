// @vitest-environment node
import { createServer, type Server } from "node:http";
import net from "node:net";
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

import { server as mswServer } from "#/mocks/node";

import { MARITIME_IPC } from "../../scripts/maritime-bridge.mjs";
import {
  RPC_METHODS,
  createMaritimeWebBridge,
  mountMaritimeWebBridge,
} from "../../scripts/maritime-web-bridge.mjs";

const KEY = "stack-session-key";

// Everything here talks to a real loopback server.
beforeAll(() => mswServer.close());
afterAll(() => mswServer.listen({ onUnhandledRequest: "bypass" }));

function fakeBridge() {
  return {
    getAuthState: vi.fn(() => ({ isSignedIn: true, isSaved: true })),
    saveApiKey: vi.fn(async () => ({ isSignedIn: true })),
    signOut: vi.fn(() => ({ isSignedIn: false })),
    getAccount: vi.fn(async () => ({ plan: "free" })),
    listAgents: vi.fn(async () => [{ id: "agent-1" }]),
    createAgent: vi.fn(async (input: unknown) => ({ input })),
    openAgent: vi.fn(async () => {
      const error = new Error("This device has no access key for that agent.");
      Object.assign(error, { status: 409 });
      throw error;
    }),
    resetAccessKey: vi.fn(),
    sleepAgent: vi.fn(),
    deleteAgent: vi.fn(),
  };
}

describe("RPC_METHODS", () => {
  it("is exactly the desktop IPC surface", () => {
    expect([...RPC_METHODS].sort()).toEqual(Object.keys(MARITIME_IPC).sort());
  });
});

describe("mountMaritimeWebBridge", () => {
  it("stays off unless MARITIME_WEB is exactly 1", () => {
    const log = vi.fn();
    expect(mountMaritimeWebBridge({ env: {}, log })).toBeNull();
    expect(
      mountMaritimeWebBridge({ env: { MARITIME_WEB: "0" }, log }),
    ).toBeNull();
    expect(log).not.toHaveBeenCalled();
    expect(
      mountMaritimeWebBridge({ env: { MARITIME_WEB: "true" }, log }),
    ).toBeNull();
    expect(log).toHaveBeenCalledWith(expect.stringContaining('is not "1"'));
  });

  it("refuses a non-loopback bind with no key", () => {
    const log = vi.fn();
    expect(
      mountMaritimeWebBridge({
        env: { MARITIME_WEB: "1" },
        host: "0.0.0.0",
        log,
      }),
    ).toBeNull();
    expect(log).toHaveBeenCalledWith(expect.stringContaining("not loopback"));
  });
});

describe("createMaritimeWebBridge", () => {
  let server: Server;
  let origin: string;
  let bridge: ReturnType<typeof fakeBridge>;

  beforeEach(async () => {
    bridge = fakeBridge();
    const web = createMaritimeWebBridge({
      bridge: bridge as never,
      key: KEY,
      host: "127.0.0.1",
      env: {},
    });
    server = createServer((req, res) => {
      if (web.handleHttp(req, res)) return;
      res.writeHead(200);
      res.end("static");
    });
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", () => resolve());
    });
    origin = `http://127.0.0.1:${(server.address() as net.AddressInfo).port}`;
  });

  afterEach(() => {
    server.closeAllConnections();
    server.close();
  });

  const rpc = (method: string, args: unknown[] = [], headers = {}) =>
    fetch(`${origin}/maritime/rpc/${method}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Session-API-Key": KEY,
        ...headers,
      },
      body: JSON.stringify({ args }),
    });

  it("answers the health probe without a key and leaves other paths alone", async () => {
    const health = await fetch(`${origin}/maritime/health`);
    expect(await health.json()).toEqual({ ok: true, authRequired: true });
    expect(await (await fetch(`${origin}/index.html`)).text()).toBe("static");
  });

  it("wants the session key on every RPC", async () => {
    const response = await rpc("listAgents", [], {
      "X-Session-API-Key": "nope",
    });
    expect(response.status).toBe(401);
    expect(bridge.listAgents).not.toHaveBeenCalled();
  });

  it("refuses cross-origin requests before touching the bridge", async () => {
    const response = await rpc("listAgents", [], {
      Origin: "https://evil.example",
    });
    expect(response.status).toBe(403);
    expect(bridge.listAgents).not.toHaveBeenCalled();
  });

  it("refuses a foreign Host on a loopback bind (DNS rebinding)", async () => {
    const response = await new Promise<number>((resolve) => {
      const req = net.connect(Number(new URL(origin).port), "127.0.0.1", () => {
        req.write(
          `POST /maritime/rpc/listAgents HTTP/1.1\r\nHost: rebound.example\r\nX-Session-API-Key: ${KEY}\r\nContent-Length: 2\r\n\r\n{}`,
        );
      });
      req.on("data", (chunk) => {
        resolve(Number(chunk.toString().split(" ")[1]));
        req.destroy();
      });
    });
    expect(response).toBe(403);
    expect(bridge.listAgents).not.toHaveBeenCalled();
  });

  it("forwards allowed methods and rejects unknown ones", async () => {
    const listed = await rpc("listAgents");
    expect(await listed.json()).toEqual({ result: [{ id: "agent-1" }] });

    const unknown = await rpc("registerIpc");
    expect(unknown.status).toBe(404);
  });

  it("takes a new agent's CORS origin from the request, not from the page", async () => {
    const response = await rpc("createAgent", [
      { name: "a", memMb: 4096, origin: "https://claimed.example" },
    ]);
    expect(await response.json()).toEqual({
      result: { input: { name: "a", memMb: 4096, origin } },
    });
  });

  it("surfaces bridge errors with their status and message", async () => {
    const response = await rpc("openAgent", ["agent-1"]);
    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({
      error: {
        message: "This device has no access key for that agent.",
        status: 409,
      },
    });
  });
});
