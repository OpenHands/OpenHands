// @vitest-environment node
import { describe, expect, it, vi } from "vitest";
import { createServer, type IncomingMessage } from "node:http";
import {
  DEPENDENCY_TRACK_PROXY_PATH_PREFIX,
  buildDependencyTrackTargetUrl,
  createDependencyTrackProxyHandler,
  defaultDependencyTrackSecretName,
  isDependencyTrackProxyRequest,
  rewriteDependencyTrackProxyPath,
} from "../../scripts/dependency-track-proxy.mjs";
import { readSessionApiKey } from "../../scripts/appwrite-proxy.mjs";

describe("dependency-track-proxy helpers", () => {
  it("matches the proxy prefix", () => {
    expect(
      isDependencyTrackProxyRequest(
        `${DEPENDENCY_TRACK_PROXY_PATH_PREFIX}/api/v1/version`,
      ),
    ).toBe(true);
    expect(isDependencyTrackProxyRequest("/api/settings")).toBe(false);
  });

  it("rewrites proxy paths to Dependency-Track API paths", () => {
    expect(
      rewriteDependencyTrackProxyPath(
        `${DEPENDENCY_TRACK_PROXY_PATH_PREFIX}/api/v1/bom`,
      ),
    ).toBe("/api/v1/bom");
    expect(rewriteDependencyTrackProxyPath(DEPENDENCY_TRACK_PROXY_PATH_PREFIX)).toBe(
      "/api/v1/version",
    );
  });

  it("builds target URLs from base URL and relative path", () => {
    expect(
      buildDependencyTrackTargetUrl(
        "https://dtrack.example.com",
        "/api/v1/version",
      ),
    ).toBe("https://dtrack.example.com/api/v1/version");
  });

  it("derives a stable secret name from workspace id", () => {
    expect(defaultDependencyTrackSecretName("ws-demo")).toBe(
      "INTEGRATION_DEPENDENCY_TRACK_API_KEY_ws_demo",
    );
  });

  it("reads the session API key header", () => {
    expect(
      readSessionApiKey({
        headers: { "x-session-api-key": "abc" },
      } as unknown as IncomingMessage),
    ).toBe("abc");
  });
});

describe("createDependencyTrackProxyHandler", () => {
  it("rejects missing session key", async () => {
    const handle = createDependencyTrackProxyHandler({
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
        `http://127.0.0.1:${address.port}${DEPENDENCY_TRACK_PROXY_PATH_PREFIX}/api/v1/version`,
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

  it("forwards X-Api-Key after resolving per-workspace config", async () => {
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
      res.end(JSON.stringify({ version: "4.12.0" }));
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
                dependencyTrack: {
                  byWorkspace: {
                    [workspaceId]: {
                      enabled: true,
                      baseUrl: `http://127.0.0.1:${upstreamAddress.port}`,
                      projectUuid: "proj-uuid-1",
                    },
                  },
                },
              },
            },
          }),
        };
      }
      return { status: 200, text: "dt-secret-key", json: () => null };
    });

    const handle = createDependencyTrackProxyHandler({
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
        `http://127.0.0.1:${address.port}${DEPENDENCY_TRACK_PROXY_PATH_PREFIX}/api/v1/version`,
        {
          headers: {
            "X-Session-API-Key": "session-1",
            "X-OpenHands-Workspace-Id": workspaceId,
          },
        },
      );
      expect(response.status).toBe(200);
      expect(upstreamHits).toHaveLength(1);
      expect(upstreamHits[0]?.apiKey).toBe("dt-secret-key");
      expect(upstreamHits[0]?.url).toBe("/api/v1/version");
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
