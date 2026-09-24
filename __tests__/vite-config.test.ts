// @vitest-environment node
import { createServer as createHttpServer, type Server } from "node:http";
import { brotliCompressSync, deflateSync, gzipSync } from "node:zlib";
import { createServer as createViteServer, type ViteDevServer } from "vite";
import viteConfig from "../vite.config";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { server as mockApiServer } from "../src/mocks/node";

afterEach(() => {
  delete process.env.BUILD_LIB;
});

describe("vite optimizeDeps", () => {
  it("prebundles core client entry dependencies", async () => {
    const config = await viteConfig({ mode: "development", command: "serve" });
    const optimizedDeps = config.optimizeDeps?.include ?? [];

    expect(optimizedDeps).toEqual(
      expect.arrayContaining([
        "react",
        "react/jsx-runtime",
        "react-dom/client",
        "react-router/dom",
      ]),
    );
  });
});

describe("vite path resolution", () => {
  it("uses Vite's native tsconfig paths support", async () => {
    const config = await viteConfig({ mode: "development", command: "serve" });

    expect(config.resolve?.tsconfigPaths).toBe(true);
    expect(config.plugins).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "vite-tsconfig-paths" }),
      ]),
    );
  });
});

describe("vite app build", () => {
  it("configures Rolldown code splitting for large vendor chunks", async () => {
    const config = await viteConfig({ mode: "production", command: "build" });
    const appBuild = config as {
      build?: {
        rolldownOptions?: {
          output?: {
            codeSplitting?: {
              groups?: Array<{
                name?: string;
                maxSize?: number;
                entriesAware?: boolean;
              }>;
            };
          };
        };
      };
    };

    expect(
      appBuild.build?.rolldownOptions?.output?.codeSplitting?.groups,
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "vendor",
          maxSize: 450 * 1024,
          entriesAware: true,
        }),
      ]),
    );
  });
});

describe("vite library build", () => {
  it("configures a dual-format preserved-module library build", async () => {
    process.env.BUILD_LIB = "true";

    const config = await viteConfig({ mode: "production", command: "build" });

    expect((config as { copyPublicDir?: boolean }).copyPublicDir).toBe(false);
    expect(config.build?.lib).toMatchObject({
      formats: ["es"],
    });
    expect(config.build?.rollupOptions?.external).toEqual(
      expect.arrayContaining(["react", "react-dom", "react-router"]),
    );
    expect(config.build?.rollupOptions?.output).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          format: "es",
          preserveModules: true,
          preserveModulesRoot: "src",
        }),
        expect.objectContaining({
          format: "cjs",
          preserveModules: true,
          preserveModulesRoot: "src",
          exports: "named",
        }),
      ]),
    );
  });
});

describe("vite runtime services metadata", () => {
  beforeAll(() => mockApiServer.close());
  const servers: Server[] = [];
  const viteServers: ViteDevServer[] = [];
  const runtimeServices = {
    mode: "dev:minimal",
    services: {
      agent_server: { url_from_agent: "http://agent.example:18000" },
    },
  };
  const serverInfo = {
    version: "1.49.2",
    usable_tools: ["terminal", "file_editor"],
    capabilities: { native_agents: true },
  };

  afterEach(async () => {
    await Promise.all(viteServers.splice(0).map((server) => server.close()));
    await Promise.all(
      servers
        .splice(0)
        .map(
          (server) =>
            new Promise<void>((resolve) => server.close(() => resolve())),
        ),
    );
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  async function startProxy({
    body = Buffer.from(JSON.stringify(serverInfo)),
    encoding = "identity",
    status = 200,
    metadata = JSON.stringify(runtimeServices),
  }: {
    body?: Buffer;
    encoding?: string;
    status?: number;
    metadata?: string;
  } = {}) {
    const requests: Array<{ url?: string; key?: string | string[] }> = [];
    const upstream = createHttpServer((req, res) => {
      requests.push({ url: req.url, key: req.headers["x-session-api-key"] });
      res.writeHead(status, {
        "content-type": "application/json",
        "content-encoding": encoding,
        "content-length": body.length,
        etag: '"sdk-version"',
      });
      res.end(body);
    });
    servers.push(upstream);
    await new Promise<void>((resolve, reject) => {
      upstream.once("error", reject);
      upstream.listen(0, "127.0.0.1", resolve);
    });
    const address = upstream.address();
    if (!address || typeof address === "string") throw new Error("No port");
    vi.stubEnv("VITE_BACKEND_HOST", `127.0.0.1:${address.port}`);
    vi.stubEnv("VITE_USE_TLS", "false");
    vi.stubEnv("VITE_RUNTIME_SERVICES_INFO", metadata);
    const config = await viteConfig({ mode: "development", command: "serve" });
    const server = await createViteServer({
      configFile: false,
      appType: "custom",
      optimizeDeps: { noDiscovery: true, include: [] },
      server: { host: "127.0.0.1", port: 0, proxy: config.server?.proxy },
    });
    viteServers.push(server);
    await server.listen();
    return { origin: server.resolvedUrls!.local[0], requests };
  }

  it.each([
    ["identity", (body: Buffer) => body],
    ["gzip", gzipSync],
    ["deflate", deflateSync],
    ["br", brotliCompressSync],
  ] as const)(
    "enriches %s SDK JSON without losing its fields or authentication",
    async (encoding, compress) => {
      const { origin, requests } = await startProxy({
        encoding,
        body: compress(Buffer.from(JSON.stringify(serverInfo))),
      });

      const response = await fetch(`${origin}server_info?check=1`, {
        headers: { "X-Session-API-Key": "test-session" },
      });

      expect(await response.json()).toEqual({
        ...serverInfo,
        runtime_services: runtimeServices,
      });
      expect(response.headers.get("content-encoding")).toBeNull();
      expect(response.headers.get("etag")).toBeNull();
      expect(response.headers.get("cache-control")).toBe("no-store");
      expect(requests).toEqual([
        { url: "/server_info?check=1", key: "test-session" },
      ]);
    },
  );

  it.each([
    { name: "no metadata", metadata: "", status: 200, payload: serverInfo },
    {
      name: "upstream authorization failure",
      metadata: JSON.stringify(runtimeServices),
      status: 401,
      payload: { detail: "Unauthorized" },
    },
    {
      name: "invalid configured metadata",
      metadata: "not-json",
      status: 200,
      payload: serverInfo,
    },
  ])(
    "preserves the upstream response with $name",
    async ({ metadata, status, payload }) => {
      vi.spyOn(console, "warn").mockImplementation(() => {});
      const { origin } = await startProxy({
        metadata,
        status,
        body: gzipSync(Buffer.from(JSON.stringify(payload))),
        encoding: "gzip",
      });

      const response = await fetch(`${origin}server_info`);

      expect(response.status).toBe(status);
      expect(await response.json()).toEqual(payload);
      expect(response.headers.get("content-encoding")).toBe("gzip");
      expect(response.headers.get("etag")).toBe('"sdk-version"');
    },
  );
});
