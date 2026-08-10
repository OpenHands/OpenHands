// @vitest-environment node
import { createHash } from "node:crypto";
import { createServer, request, type IncomingMessage } from "node:http";
import { describe, expect, it, vi } from "vitest";
import {
  createDesktopProxyHandler,
  DESKTOP_IFRAME_PATH,
  DESKTOP_PROXY_PATH_PREFIX,
  isDesktopProxyRequest,
  isDesktopStaticAssetPath,
  isVncServerInstalled,
  readCookie,
  resolveDesktopDir,
  rewriteDesktopProxyPath,
} from "../../scripts/desktop-proxy.mjs";

describe("desktop-proxy helpers", () => {
  it("matches the proxy prefix", () => {
    expect(isDesktopProxyRequest(`${DESKTOP_PROXY_PATH_PREFIX}/`)).toBe(true);
    expect(isDesktopProxyRequest(`${DESKTOP_PROXY_PATH_PREFIX}/start`)).toBe(
      true,
    );
    expect(isDesktopProxyRequest("/api/settings")).toBe(false);
  });

  it("rewrites proxy paths onto the KasmVNC root", () => {
    expect(rewriteDesktopProxyPath(DESKTOP_PROXY_PATH_PREFIX)).toBe("/");
    expect(rewriteDesktopProxyPath(`${DESKTOP_PROXY_PATH_PREFIX}/`)).toBe("/");
    expect(
      rewriteDesktopProxyPath(`${DESKTOP_PROXY_PATH_PREFIX}/index.html`),
    ).toBe("/index.html");
  });

  it("identifies anonymous-safe static asset paths", () => {
    expect(
      isDesktopStaticAssetPath(`${DESKTOP_PROXY_PATH_PREFIX}/main.bundle.js`),
    ).toBe(true);
    expect(
      isDesktopStaticAssetPath(
        `${DESKTOP_PROXY_PATH_PREFIX}/assets/webutil.css`,
      ),
    ).toBe(true);
    expect(
      isDesktopStaticAssetPath(`${DESKTOP_PROXY_PATH_PREFIX}/index.html`),
    ).toBe(false);
    expect(
      isDesktopStaticAssetPath(`${DESKTOP_PROXY_PATH_PREFIX}/websockify`),
    ).toBe(false);
  });

  it("reads cookies from the Cookie header", () => {
    expect(
      readCookie(
        {
          headers: {
            cookie: "a=1; agent-canvas-desktop-auth=secret%2Bkey; b=2",
          },
        } as unknown as IncomingMessage,
        "agent-canvas-desktop-auth",
      ),
    ).toBe("secret+key");
    expect(
      readCookie({ headers: {} } as unknown as IncomingMessage, "x"),
    ).toBeNull();
  });

  it("resolves a desktop dir that contains start-desktop.sh", () => {
    const dir = resolveDesktopDir();
    expect(dir.length).toBeGreaterThan(0);
  });

  it("reports whether vncserver is on PATH", () => {
    expect(typeof isVncServerInstalled()).toBe("boolean");
  });
});

describe("createDesktopProxyHandler", () => {
  it("rejects start without session key", async () => {
    const handle = createDesktopProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
      desktopAvailableImpl: () => true,
      healthCheckImpl: async () => false,
      fetchImpl: vi.fn(),
    });
    const server = createServer((req, res) => {
      void handle(req, res);
    });
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("expected TCP address");
    }
    try {
      const response = await fetch(
        `http://127.0.0.1:${address.port}${DESKTOP_PROXY_PATH_PREFIX}/start`,
        { method: "POST" },
      );
      expect(response.status).toBe(401);
      const body = (await response.json()) as { detail: string };
      expect(body.detail).toMatch(/X-Session-API-Key/);
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });

  it("reports unavailable when desktop scripts are missing", async () => {
    const handle = createDesktopProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
      desktopAvailableImpl: () => false,
      healthCheckImpl: async () => false,
      fetchImpl: vi.fn(),
    });
    const server = createServer((req, res) => {
      void handle(req, res);
    });
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("expected TCP address");
    }
    try {
      const response = await fetch(
        `http://127.0.0.1:${address.port}${DESKTOP_PROXY_PATH_PREFIX}/status`,
      );
      expect(response.status).toBe(200);
      const body = (await response.json()) as {
        unavailable: boolean;
        ready: boolean;
      };
      expect(body.unavailable).toBe(true);
      expect(body.ready).toBe(false);
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });

  it("starts the desktop and sets the auth cookie when session is valid", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({}),
    }));
    const handle = createDesktopProxyHandler({
      agentServerUrl: "http://127.0.0.1:18000",
      desktopAvailableImpl: () => true,
      healthCheckImpl: async () => true,
      fetchImpl: fetchImpl as unknown as typeof fetch,
      spawnImpl: vi.fn() as never,
    });
    const server = createServer((req, res) => {
      void handle(req, res);
    });
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("expected TCP address");
    }
    try {
      const response = await fetch(
        `http://127.0.0.1:${address.port}${DESKTOP_PROXY_PATH_PREFIX}/start`,
        {
          method: "POST",
          headers: { "X-Session-API-Key": "session-1" },
        },
      );
      expect(response.status).toBe(200);
      const body = (await response.json()) as { ready: boolean; url: string };
      expect(body.ready).toBe(true);
      expect(body.url).toBe(DESKTOP_IFRAME_PATH);
      const setCookie = response.headers.get("set-cookie") ?? "";
      expect(setCookie).toContain("agent-canvas-desktop-auth=");
      expect(setCookie).toContain("HttpOnly");
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });

  it("rejects websocket upgrades without auth", async () => {
    const handle = createDesktopProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
      desktopAvailableImpl: () => true,
      healthCheckImpl: async () => true,
      fetchImpl: vi.fn(),
    });

    const req = {
      url: `${DESKTOP_PROXY_PATH_PREFIX}/websockify`,
      headers: {},
    } as unknown as IncomingMessage;
    const chunks: Buffer[] = [];
    const socket = {
      write: (data: string | Buffer) => {
        chunks.push(Buffer.from(data));
      },
      destroy: vi.fn(),
    };

    const handled = await handle.handleUpgrade(
      req,
      socket as never,
      Buffer.alloc(0),
    );
    expect(handled).toBe(true);
    expect(Buffer.concat(chunks).toString()).toContain("401");
    expect(socket.destroy).toHaveBeenCalled();
  });

  it("probes KasmVNC-style Connection:close responses without crashing Node", async () => {
    // Regression: undici fetch + unread body + Connection:close used to raise
    // `assert(!this.paused)` and kill ingress when Desktop status/start ran.
    const payload = Buffer.alloc(64 * 1024, 0x61);
    const vnc = createServer((req, res) => {
      if (req.url?.startsWith("/index.html")) {
        res.writeHead(200, {
          "Content-Type": "text/html",
          "Content-Length": payload.length,
          Connection: "close",
          "Cross-Origin-Embedder-Policy": "require-corp",
        });
        res.end(payload);
        return;
      }
      res.writeHead(404);
      res.end();
    });
    await new Promise<void>((resolve) => {
      vnc.listen(0, "127.0.0.1", resolve);
    });
    const vncAddr = vnc.address();
    if (!vncAddr || typeof vncAddr === "string") {
      throw new Error("expected TCP address");
    }

    const handle = createDesktopProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
      vncPort: vncAddr.port,
      desktopAvailableImpl: () => true,
      fetchImpl: vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({}),
      })) as unknown as typeof fetch,
    });
    const server = createServer((req, res) => {
      void handle(req, res);
    });
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("expected TCP address");
    }

    try {
      for (let i = 0; i < 5; i += 1) {
        const response = await fetch(
          `http://127.0.0.1:${address.port}${DESKTOP_PROXY_PATH_PREFIX}/status`,
        );
        expect(response.status).toBe(200);
        const body = (await response.json()) as { ready: boolean };
        expect(body.ready).toBe(true);
      }

      const start = await fetch(
        `http://127.0.0.1:${address.port}${DESKTOP_PROXY_PATH_PREFIX}/start`,
        {
          method: "POST",
          headers: { "X-Session-API-Key": "session-1" },
        },
      );
      expect(start.status).toBe(200);
      const cookie = start.headers.get("set-cookie") ?? "";
      expect(cookie).toContain("agent-canvas-desktop-auth=");

      const page = await fetch(
        `http://127.0.0.1:${address.port}${DESKTOP_PROXY_PATH_PREFIX}/index.html`,
        { headers: { Cookie: cookie.split(";")[0] ?? "" } },
      );
      expect(page.status).toBe(200);
      expect(page.headers.get("cross-origin-embedder-policy")).toBeNull();
      expect(page.headers.get("access-control-allow-origin")).toBeTruthy();
      const html = await page.text();
      expect(html.length).toBe(payload.length);
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
      await new Promise<void>((resolve, reject) => {
        vnc.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });

  it("sets the auth cookie on status when a valid session key is sent", async () => {
    const handle = createDesktopProxyHandler({
      agentServerUrl: "http://127.0.0.1:18000",
      desktopAvailableImpl: () => true,
      healthCheckImpl: async () => true,
      fetchImpl: vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({}),
      })) as unknown as typeof fetch,
    });
    const server = createServer((req, res) => {
      void handle(req, res);
    });
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("expected TCP address");
    }
    try {
      const response = await fetch(
        `http://127.0.0.1:${address.port}${DESKTOP_PROXY_PATH_PREFIX}/status`,
        { headers: { "X-Session-API-Key": "session-status" } },
      );
      expect(response.status).toBe(200);
      const setCookie = response.headers.get("set-cookie") ?? "";
      expect(setCookie).toContain("agent-canvas-desktop-auth=");
      expect(setCookie).toContain("session-status");
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });

  it("proxies websockify upgrades with the headers KasmVNC requires", async () => {
    const seen: Record<string, string | string[] | undefined>[] = [];
    const vnc = createServer(() => {
      /* upgrade-only upstream */
    });
    vnc.on("upgrade", (req, socket) => {
      seen.push({ ...req.headers });
      const key = String(req.headers["sec-websocket-key"] ?? "");
      const accept = createHash("sha1")
        .update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
        .digest("base64");
      socket.write(
        "HTTP/1.1 101 Switching Protocols\r\n" +
          "Upgrade: websocket\r\n" +
          "Connection: Upgrade\r\n" +
          `Sec-WebSocket-Accept: ${accept}\r\n` +
          "Sec-WebSocket-Protocol: binary\r\n\r\n",
      );
      socket.end();
    });
    await new Promise<void>((resolve) => {
      vnc.listen(0, "127.0.0.1", resolve);
    });
    const vncAddr = vnc.address();
    if (!vncAddr || typeof vncAddr === "string") {
      throw new Error("expected TCP address");
    }

    const handle = createDesktopProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
      vncPort: vncAddr.port,
      desktopAvailableImpl: () => true,
      healthCheckImpl: async () => true,
      fetchImpl: vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({}),
      })) as unknown as typeof fetch,
    });

    const server = createServer((req, res) => {
      void handle(req, res);
    });
    server.on("upgrade", (req, socket, head) => {
      void handle.handleUpgrade(req, socket, head);
    });
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("expected TCP address");
    }

    try {
      const start = await fetch(
        `http://127.0.0.1:${address.port}${DESKTOP_PROXY_PATH_PREFIX}/start`,
        {
          method: "POST",
          headers: { "X-Session-API-Key": "session-ws" },
        },
      );
      expect(start.status).toBe(200);
      const cookie = (start.headers.get("set-cookie") ?? "").split(";")[0] ?? "";

      await new Promise<void>((resolve, reject) => {
        const req = request({
          hostname: "127.0.0.1",
          port: address.port,
          path: `${DESKTOP_PROXY_PATH_PREFIX}/websockify`,
          method: "GET",
          headers: {
            Connection: "Upgrade",
            Upgrade: "websocket",
            "Sec-WebSocket-Version": "13",
            "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
            "Sec-WebSocket-Protocol": "binary",
            Origin: "http://127.0.0.1:3000",
            Cookie: cookie,
          },
        });
        req.on("upgrade", (_res, socket) => {
          socket.destroy();
          resolve();
        });
        req.on("response", (res) => {
          reject(new Error(`expected 101, got ${res.statusCode}`));
        });
        req.on("error", reject);
        req.end();
      });

      expect(seen.length).toBeGreaterThan(0);
      const headers = seen[0] ?? {};
      expect(String(headers.host)).toContain(String(vncAddr.port));
      expect(headers["sec-websocket-origin"]).toBe("http://127.0.0.1:3000");
      expect(headers["sec-websocket-protocol"]).toBe("binary");
      expect(String(headers.authorization ?? "")).toMatch(/^Basic /);
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
      await new Promise<void>((resolve, reject) => {
        vnc.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });
});
