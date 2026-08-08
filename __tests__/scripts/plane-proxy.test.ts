// @vitest-environment node
import { describe, expect, it, vi } from "vitest";
import { createServer, type IncomingMessage } from "node:http";
import {
  PLANE_PROXY_PATH_PREFIX,
  buildPlaneTargetUrl,
  buildPlaneTestPath,
  createPlaneProxyHandler,
  isPlaneProxyRequest,
  readSessionApiKey,
  rewritePlaneProxyPath,
} from "../../scripts/plane-proxy.mjs";

describe("plane-proxy helpers", () => {
  it("matches the proxy prefix", () => {
    expect(isPlaneProxyRequest(`${PLANE_PROXY_PATH_PREFIX}/test`)).toBe(true);
    expect(isPlaneProxyRequest("/api/settings")).toBe(false);
  });

  it("rewrites proxy paths", () => {
    expect(rewritePlaneProxyPath(`${PLANE_PROXY_PATH_PREFIX}/test`)).toBe(
      "/test",
    );
    expect(rewritePlaneProxyPath(PLANE_PROXY_PATH_PREFIX)).toBe("");
  });

  it("builds Plane target URLs", () => {
    expect(
      buildPlaneTargetUrl(
        "https://plane.example.com",
        "/api/v1/workspaces/heimdall/projects/p1/",
      ),
    ).toBe("https://plane.example.com/api/v1/workspaces/heimdall/projects/p1/");
  });

  it("builds test paths for project and optional module", () => {
    expect(
      buildPlaneTestPath("https://plane.example.com", "heimdall", "proj-1"),
    ).toBe("/api/v1/workspaces/heimdall/projects/proj-1/");
    expect(
      buildPlaneTestPath(
        "https://plane.example.com",
        "heimdall",
        "proj-1",
        "mod-1",
      ),
    ).toBe("/api/v1/workspaces/heimdall/projects/proj-1/modules/mod-1/");
  });

  it("reads the session API key header", () => {
    expect(
      readSessionApiKey({
        headers: { "x-session-api-key": "abc" },
      } as unknown as IncomingMessage),
    ).toBe("abc");
    expect(
      readSessionApiKey({ headers: {} } as unknown as IncomingMessage),
    ).toBeNull();
  });
});

describe("createPlaneProxyHandler", () => {
  it("rejects missing session key", async () => {
    const handle = createPlaneProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
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
        `http://127.0.0.1:${address.port}${PLANE_PROXY_PATH_PREFIX}/test`,
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

  it("rejects missing workspace id", async () => {
    const handle = createPlaneProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
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
        `http://127.0.0.1:${address.port}${PLANE_PROXY_PATH_PREFIX}/test`,
        { headers: { "X-Session-API-Key": "session-1" } },
      );
      expect(response.status).toBe(400);
      const body = (await response.json()) as { detail: string };
      expect(body.detail).toMatch(/X-OpenHands-Workspace-Id/);
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });

  it("forwards X-API-Key after resolving per-workspace config", async () => {
    const workspaceId = "ws-demo";
    const upstreamHits: Array<{
      method?: string;
      url?: string;
      apiKey?: string | string[];
    }> = [];
    const upstream = createServer((req, res) => {
      upstreamHits.push({
        method: req.method,
        url: req.url,
        apiKey: req.headers["x-api-key"],
      });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ id: "proj-1", name: "Demo" }));
    });
    await new Promise<void>((resolve) => {
      upstream.listen(0, "127.0.0.1", resolve);
    });
    const upstreamAddress = upstream.address();
    if (!upstreamAddress || typeof upstreamAddress === "string") {
      throw new Error("expected TCP address");
    }
    const baseUrl = `http://127.0.0.1:${upstreamAddress.port}`;

    const fetchImpl = vi.fn(
      async (
        _agentServerUrl: string,
        _sessionApiKey: string,
        path: string,
      ) => {
        if (path === "/api/settings") {
          return {
            status: 200,
            text: "",
            json: () => ({
              misc_settings: {
                integrations: {
                  plane: {
                    byWorkspace: {
                      [workspaceId]: {
                        enabled: true,
                        baseUrl,
                        workspaceSlug: "heimdall",
                        projectId: "proj-1",
                      },
                    },
                  },
                },
              },
            }),
          };
        }
        if (path.startsWith("/api/settings/secrets/")) {
          return {
            status: 200,
            text: "plane-secret-key",
            json: () => null,
          };
        }
        return { status: 404, text: "", json: () => null };
      },
    );

    const handle = createPlaneProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
      fetchImpl,
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
        `http://127.0.0.1:${address.port}${PLANE_PROXY_PATH_PREFIX}/test`,
        {
          headers: {
            "X-Session-API-Key": "session-1",
            "X-OpenHands-Workspace-Id": workspaceId,
          },
        },
      );
      expect(response.status).toBe(200);
      expect(upstreamHits).toHaveLength(1);
      expect(upstreamHits[0].method).toBe("GET");
      expect(upstreamHits[0].url).toBe(
        "/api/v1/workspaces/heimdall/projects/proj-1/",
      );
      expect(upstreamHits[0].apiKey).toBe("plane-secret-key");
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
      await new Promise<void>((resolve, reject) => {
        upstream.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });
});
