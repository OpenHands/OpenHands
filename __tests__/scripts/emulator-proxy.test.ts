// @vitest-environment node
import { createServer } from "node:http";
import { describe, expect, it, vi } from "vitest";
import {
  createEmulatorProxyHandler,
  EMULATOR_IFRAME_PATH,
  EMULATOR_PROXY_PATH_PREFIX,
  isEmulatorProxyRequest,
  isEmulatorStaticAssetPath,
  resolveEmulatorUpstreamUrl,
  rewriteEmulatorProxyPath,
} from "../../scripts/emulator-proxy.mjs";

describe("emulator-proxy helpers", () => {
  it("matches the proxy prefix", () => {
    expect(isEmulatorProxyRequest(`${EMULATOR_PROXY_PATH_PREFIX}/`)).toBe(true);
    expect(isEmulatorProxyRequest(`${EMULATOR_PROXY_PATH_PREFIX}/start`)).toBe(
      true,
    );
    expect(isEmulatorProxyRequest("/api/settings")).toBe(false);
  });

  it("rewrites proxy paths", () => {
    expect(rewriteEmulatorProxyPath(EMULATOR_PROXY_PATH_PREFIX)).toBe("/");
    expect(
      rewriteEmulatorProxyPath(`${EMULATOR_PROXY_PATH_PREFIX}/index.html`),
    ).toBe("/index.html");
  });

  it("identifies static assets", () => {
    expect(
      isEmulatorStaticAssetPath(`${EMULATOR_PROXY_PATH_PREFIX}/app.js`),
    ).toBe(true);
    expect(
      isEmulatorStaticAssetPath(`${EMULATOR_PROXY_PATH_PREFIX}/index.html`),
    ).toBe(false);
  });

  it("resolves upstream from env/runtime_services", () => {
    expect(resolveEmulatorUpstreamUrl({})).toBeNull();
    expect(
      resolveEmulatorUpstreamUrl({
        upstreamUrl: "http://127.0.0.1:6080/",
      }),
    ).toBe("http://127.0.0.1:6080");
    expect(
      resolveEmulatorUpstreamUrl({
        runtimeServicesInfo: {
          services: {
            android_emulator: { url_from_agent: "http://emulator:6080" },
          },
        },
      }),
    ).toBe("http://emulator:6080");
  });
});

describe("createEmulatorProxyHandler auth (AC-192-6)", () => {
  it("rejects start without session key", async () => {
    const handle = createEmulatorProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
      upstreamUrl: "http://127.0.0.1:6080",
      healthCheckImpl: async () => true,
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
        `http://127.0.0.1:${address.port}${EMULATOR_PROXY_PATH_PREFIX}/start`,
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

  it("reports unavailable when upstream is missing", async () => {
    const handle = createEmulatorProxyHandler({
      agentServerUrl: "http://127.0.0.1:9",
      upstreamAvailableImpl: () => false,
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
        `http://127.0.0.1:${address.port}${EMULATOR_PROXY_PATH_PREFIX}/status`,
      );
      expect(response.status).toBe(200);
      const body = (await response.json()) as {
        unavailable: boolean;
        url: string;
      };
      expect(body.unavailable).toBe(true);
      expect(body.url).toBe(EMULATOR_IFRAME_PATH);
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });
});
