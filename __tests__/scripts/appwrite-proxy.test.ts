// @vitest-environment node
import { describe, expect, it, vi } from "vitest";
import { createServer, type IncomingMessage } from "node:http";
import {
  APPWRITE_PROXY_PATH_PREFIX,
  buildAppwriteTargetUrl,
  createAppwriteProxyHandler,
  isAppwriteProxyRequest,
  readSessionApiKey,
  rewriteAppwriteProxyPath,
} from "../../scripts/appwrite-proxy.mjs";

describe("appwrite-proxy helpers", () => {
  it("matches the proxy prefix", () => {
    expect(
      isAppwriteProxyRequest(`${APPWRITE_PROXY_PATH_PREFIX}/v1/databases`),
    ).toBe(true);
    expect(isAppwriteProxyRequest("/api/settings")).toBe(false);
  });

  it("rewrites proxy paths", () => {
    expect(
      rewriteAppwriteProxyPath(`${APPWRITE_PROXY_PATH_PREFIX}/v1/databases`),
    ).toBe("/v1/databases");
    expect(rewriteAppwriteProxyPath(APPWRITE_PROXY_PATH_PREFIX)).toBe("");
  });

  it("avoids double /v1 when joining endpoint and path", () => {
    expect(
      buildAppwriteTargetUrl("https://cloud.appwrite.io/v1", "/v1/databases"),
    ).toBe("https://cloud.appwrite.io/v1/databases");
    expect(buildAppwriteTargetUrl("https://example.com/v1", "/health")).toBe(
      "https://example.com/v1/health",
    );
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

describe("createAppwriteProxyHandler", () => {
  it("rejects missing session key", async () => {
    const handle = createAppwriteProxyHandler({
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
        `http://127.0.0.1:${address.port}${APPWRITE_PROXY_PATH_PREFIX}/v1/health`,
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
    const handle = createAppwriteProxyHandler({
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
        `http://127.0.0.1:${address.port}${APPWRITE_PROXY_PATH_PREFIX}/v1/health`,
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

  it("forwards AppWrite headers after resolving per-workspace config", async () => {
    const workspaceId = "ws-demo";
    const upstreamHits: Array<{
      method?: string;
      url?: string;
      project?: string | string[];
      key?: string | string[];
    }> = [];
    const upstream = createServer((req, res) => {
      upstreamHits.push({
        method: req.method,
        url: req.url,
        project: req.headers["x-appwrite-project"],
        key: req.headers["x-appwrite-key"],
      });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
    });
    await new Promise<void>((resolve) => {
      upstream.listen(0, "127.0.0.1", resolve);
    });
    const upstreamAddress = upstream.address();
    if (!upstreamAddress || typeof upstreamAddress === "string") {
      throw new Error("expected TCP address");
    }

    const fetchImpl = vi.fn(async (_base: string, _key: string, path: string) => {
      if (path === "/api/settings") {
        return {
          status: 200,
          text: "",
          json: () => ({
            misc_settings: {
              integrations: {
                appwrite: {
                  byWorkspace: {
                    [workspaceId]: {
                      enabled: true,
                      endpoint: `http://127.0.0.1:${upstreamAddress.port}/v1`,
                      projectId: "proj-1",
                    },
                  },
                },
              },
            },
          }),
        };
      }
      return { status: 200, text: "secret-key", json: () => null };
    });

    const handle = createAppwriteProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
      fetchImpl,
      cacheTtlMs: 0,
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
        `http://127.0.0.1:${address.port}${APPWRITE_PROXY_PATH_PREFIX}/v1/databases`,
        {
          headers: {
            "X-Session-API-Key": "session-1",
            "X-OpenHands-Workspace-Id": workspaceId,
          },
        },
      );
      expect(response.status).toBe(200);
      expect(upstreamHits).toHaveLength(1);
      expect(upstreamHits[0]?.project).toBe("proj-1");
      expect(upstreamHits[0]?.key).toBe("secret-key");
      expect(upstreamHits[0]?.url).toBe("/v1/databases");
      expect(fetchImpl).toHaveBeenCalledWith(
        "http://127.0.0.1:9",
        "session-1",
        expect.stringContaining("INTEGRATION_APPWRITE_API_KEY_ws_demo"),
        expect.objectContaining({ responseType: "text" }),
      );
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
