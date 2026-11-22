import {
  createServer,
  request,
  type IncomingMessage,
  type Server,
} from "node:http";
import {
  connect as netConnect,
  type Server as NetServer,
  type Socket,
} from "node:net";
import type { Duplex } from "node:stream";
import { createServer as createTlsServer } from "node:tls";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  parseArgs,
  serializeForInlineScript,
  startStaticServer,
} from "../../scripts/static-server.mjs";

describe("static-server.mjs", () => {
  const servers: NetServer[] = [];
  const tempDirs: string[] = [];

  afterEach(async () => {
    await Promise.all(
      servers.splice(0).map(
        (server) =>
          new Promise<void>((resolve) => {
            server.close(() => resolve());
          }),
      ),
    );

    for (const dir of tempDirs.splice(0)) {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  async function startServer(
    dir: string,
    overrides: Partial<Parameters<typeof startStaticServer>[0]> = {},
  ) {
    const server = await startStaticServer({
      port: 0,
      host: "127.0.0.1",
      dir,
      routes: {},
      ...overrides,
    });
    servers.push(server);

    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("Static server did not bind to a TCP port");
    }

    return `http://127.0.0.1:${address.port}`;
  }

  async function startHttpServer(server: NetServer) {
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => {
        server.off("error", reject);
        resolve();
      });
    });
    servers.push(server);
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("Server did not bind to a TCP port");
    }
    return `http://127.0.0.1:${address.port}`;
  }

  async function getJson(url: string) {
    return new Promise<{ status: number; body: unknown }>((resolve, reject) => {
      const req = request(url, { method: "GET" }, (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          body += chunk;
        });
        res.on("end", () => {
          try {
            resolve({
              status: res.statusCode ?? 0,
              body: JSON.parse(body),
            });
          } catch (error) {
            reject(error);
          }
        });
      });
      req.on("error", reject);
      req.end();
    });
  }

  async function getText(url: string, headers?: Record<string, string>) {
    return new Promise<{ status: number; body: string }>((resolve, reject) => {
      const req = request(url, { method: "GET", headers }, (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          body += chunk;
        });
        res.on("end", () => {
          resolve({
            status: res.statusCode ?? 0,
            body,
          });
        });
      });
      req.on("error", reject);
      req.end();
    });
  }

  describe("parseArgs", () => {
    it("defaults host to IPv4 loopback", () => {
      expect(parseArgs([]).host).toBe("127.0.0.1");
    });

    it("defaults sessionApiKey to null", () => {
      const config = parseArgs([]);
      expect(config.sessionApiKey).toBeNull();
    });

    it("parses --session-api-key", () => {
      const config = parseArgs(["--session-api-key", "my-test-key"]);
      expect(config.sessionApiKey).toBe("my-test-key");
    });

    it("treats empty string as null for session key", () => {
      const config = parseArgs(["--session-api-key", ""]);
      expect(config.sessionApiKey).toBeNull();
    });

    it("falls back to the SESSION_API_KEY env var when no flag is given", () => {
      const previous = process.env.SESSION_API_KEY;
      try {
        process.env.SESSION_API_KEY = "env-key";
        const config = parseArgs([]);
        expect(config.sessionApiKey).toBe("env-key");
      } finally {
        if (previous === undefined) {
          delete process.env.SESSION_API_KEY;
        } else {
          process.env.SESSION_API_KEY = previous;
        }
      }
    });

    it("prefers --session-api-key over the SESSION_API_KEY env var", () => {
      const previous = process.env.SESSION_API_KEY;
      try {
        process.env.SESSION_API_KEY = "env-key";
        const config = parseArgs(["--session-api-key", "argv-key"]);
        expect(config.sessionApiKey).toBe("argv-key");
      } finally {
        if (previous === undefined) {
          delete process.env.SESSION_API_KEY;
        } else {
          process.env.SESSION_API_KEY = previous;
        }
      }
    });

    it("ignores the SESSION_API_KEY env var in public mode (--auth-required)", () => {
      const previous = process.env.SESSION_API_KEY;
      try {
        process.env.SESSION_API_KEY = "env-key";
        const config = parseArgs(["--auth-required"]);
        expect(config.sessionApiKey).toBeNull();
        expect(config.authRequired).toBe(true);
      } finally {
        if (previous === undefined) {
          delete process.env.SESSION_API_KEY;
        } else {
          process.env.SESSION_API_KEY = previous;
        }
      }
    });

    it("defaults runtimeServicesInfo to null", () => {
      const config = parseArgs([]);
      expect(config.runtimeServicesInfo).toBeNull();
    });

    it("defaults lockToCloud to null", () => {
      const config = parseArgs([]);
      expect(config.lockToCloud).toBeNull();
    });

    it("parses --lock-to-cloud", () => {
      const config = parseArgs([
        "--lock-to-cloud",
        "https://cloud.example.com",
      ]);
      expect(config.lockToCloud).toBe("https://cloud.example.com");
    });

    it("treats empty string as null for lockToCloud", () => {
      const config = parseArgs(["--lock-to-cloud", ""]);
      expect(config.lockToCloud).toBeNull();
    });

    it("defaults disableTelemetry to false", () => {
      const config = parseArgs([], {});
      expect(config.disableTelemetry).toBe(false);
    });

    it("parses --disable-telemetry", () => {
      const config = parseArgs(["--disable-telemetry"], {});
      expect(config.disableTelemetry).toBe(true);
    });

    it("enables disableTelemetry from AGENT_CANVAS_DISABLE_TELEMETRY=1", () => {
      const config = parseArgs([], { AGENT_CANVAS_DISABLE_TELEMETRY: "1" });
      expect(config.disableTelemetry).toBe(true);
    });

    it("enables disableTelemetry from AGENT_CANVAS_DISABLE_TELEMETRY=true", () => {
      const config = parseArgs([], { AGENT_CANVAS_DISABLE_TELEMETRY: "true" });
      expect(config.disableTelemetry).toBe(true);
    });

    it("ignores a falsy AGENT_CANVAS_DISABLE_TELEMETRY value", () => {
      const config = parseArgs([], { AGENT_CANVAS_DISABLE_TELEMETRY: "0" });
      expect(config.disableTelemetry).toBe(false);
    });

    it("defaults basePath to root", () => {
      const config = parseArgs([]);
      expect(config.basePath).toBe("/");
    });

    it("parses and normalizes --base-path", () => {
      const config = parseArgs(["--base-path", "canvas/"]);
      expect(config.basePath).toBe("/canvas");
    });

    it("treats empty string as root for basePath", () => {
      const config = parseArgs(["--base-path", ""]);
      expect(config.basePath).toBe("/");
    });

    it("parses --runtime-services-info", () => {
      const json = '{"mode":"docker"}';
      const config = parseArgs(["--runtime-services-info", json]);
      expect(config.runtimeServicesInfo).toBe(json);
    });

    it("treats empty string as null for runtime services info", () => {
      const config = parseArgs(["--runtime-services-info", ""]);
      expect(config.runtimeServicesInfo).toBeNull();
    });
  });

  describe("serializeForInlineScript", () => {
    it("escapes '<' and '>' to prevent breaking out of script tags", () => {
      const input = "</script><script>alert(1)</script>";
      const result = serializeForInlineScript(input);
      expect(result).not.toContain("<");
      expect(result).not.toContain(">");
      expect(result).toBe(
        '"\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e"',
      );
    });

    it("escapes line and paragraph separators U+2028 and U+2029", () => {
      const input = "line1\u2028line2\u2029line3";
      const result = serializeForInlineScript(input);
      expect(result).toContain("\\u2028");
      expect(result).toContain("\\u2029");
      expect(result).not.toContain("\u2028");
      expect(result).not.toContain("\u2029");
    });

    it("serializes complex objects properly with escaping", () => {
      const input = { key: "<test>", count: 42 };
      const result = serializeForInlineScript(input);
      expect(result).toBe('{"key":"\\u003ctest\\u003e","count":42}');
    });
  });

  describe("--vscode-base-path", () => {
    it("parses the prefix and normalizes a trailing slash", () => {
      const config = parseArgs([
        "--route",
        "/vscode=http://127.0.0.1:8001",
        "--vscode-base-path",
        "/vscode/",
      ]);
      expect(config.vscodeBasePath).toBe("/vscode");
    });

    it("defaults to null so an origin advertises nothing unless asked", () => {
      expect(parseArgs([]).vscodeBasePath).toBeNull();
    });

    it("rejects a value that is not a path", () => {
      expect(() => parseArgs(["--vscode-base-path", "vscode"])).toThrow(
        /must start with '\//,
      );
    });

    // The guard that makes this flag trustworthy: advertising the editor and
    // routing it are the same decision, so a prefix with no route behind it
    // must not start. Without it, the frontend would render the control and
    // the navigation would fall through to the SPA — the exact bug the flag
    // exists to prevent.
    it("refuses to start when the advertised prefix has no matching route", () => {
      const exit = vi.spyOn(process, "exit").mockImplementation((() => {
        throw new Error("process.exit");
      }) as never);
      const error = vi.spyOn(console, "error").mockImplementation(() => {});

      try {
        expect(() =>
          parseArgs([
            "--route",
            "/api=http://127.0.0.1:8000",
            "--vscode-base-path",
            "/vscode",
          ]),
        ).toThrow("process.exit");
        expect(exit).toHaveBeenCalledWith(1);
        expect(error.mock.calls[0][0]).toContain("has no matching --route");
      } finally {
        exit.mockRestore();
        error.mockRestore();
      }
    });
  });

  describe("vscode base path injection", () => {
    async function startServerAdvertising(
      dir: string,
      vscodeBasePath: string | null,
    ) {
      const server = await startStaticServer({
        port: 0,
        host: "127.0.0.1",
        dir,
        routes: {},
        vscodeBasePath,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") {
        throw new Error("Static server did not bind to a TCP port");
      }
      return `http://127.0.0.1:${address.port}`;
    }

    function makeBuildDir() {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );
      return buildDir;
    }

    // Read by getOriginVSCodeBasePath() in src/utils/vscode-origin.ts, which
    // is what lets the frontend tell "this server has an editor" apart from
    // "this origin can reach it".
    it("exposes the prefix on window.__AGENT_CANVAS_VSCODE_BASE_PATH__", async () => {
      const origin = await startServerAdvertising(makeBuildDir(), "/vscode");
      const body = await (await fetch(`${origin}/`)).text();

      expect(body).toContain(
        'window.__AGENT_CANVAS_VSCODE_BASE_PATH__="/vscode"',
      );
    });

    // Public mode: the same document, served without the advertisement, is
    // what hides the control on an origin that has no editor route.
    it("injects nothing when the origin serves no editor", async () => {
      const origin = await startServerAdvertising(makeBuildDir(), null);
      const body = await (await fetch(`${origin}/`)).text();

      expect(body).not.toContain("__AGENT_CANVAS_VSCODE_BASE_PATH__");
    });
  });

  describe("runtime services info exposure", () => {
    async function startServerWithRuntimeInfo(
      dir: string,
      runtimeServicesInfo: string,
    ) {
      const server = await startStaticServer({
        port: 0,
        host: "127.0.0.1",
        dir,
        routes: {},
        runtimeServicesInfo,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") {
        throw new Error("Static server did not bind to a TCP port");
      }
      return `http://127.0.0.1:${address.port}`;
    }

    // Legacy compatibility for older Docker / published-binary frontend
    // bundles, which read runtime services from this injected window global.
    // New bundles read /server_info.runtime_services instead.
    it("exposes the JSON on window.__AGENT_CANVAS_RUNTIME_SERVICES_INFO__", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const info = JSON.stringify({
        mode: "docker",
        services: {
          agent_server: { url_from_agent: "http://127.0.0.1:18000" },
        },
      });
      const origin = await startServerWithRuntimeInfo(buildDir, info);
      const body = await (await fetch(`${origin}/`)).text();

      expect(body).toContain("window.__AGENT_CANVAS_RUNTIME_SERVICES_INFO__");
      // Stored as a JSON *string* (note the escaped quotes) so older browser
      // code can JSON.parse it.
      expect(body).toContain('\\"mode\\"');
      expect(body).toContain("docker");
    });

    it("does not inject when runtimeServicesInfo is null", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const server = await startStaticServer({
        port: 0,
        host: "127.0.0.1",
        dir: buildDir,
        routes: {},
        runtimeServicesInfo: null,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") throw new Error("No port");
      const origin = `http://127.0.0.1:${(address as { port: number }).port}`;

      const body = await (await fetch(`${origin}/`)).text();
      expect(body).not.toContain("__AGENT_CANVAS_RUNTIME_SERVICES_INFO__");
    });

    it("adds runtime_services to proxied /server_info", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const upstreamOrigin = await startHttpServer(
        createServer((_req, res) => {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ version: "1.48.0" }));
        }),
      );
      const runtimeServicesInfo = JSON.stringify({
        mode: "docker",
        services: {
          agent_server: { url_from_agent: "http://127.0.0.1:18000" },
        },
      });

      const server = await startStaticServer({
        port: 0,
        host: "127.0.0.1",
        dir: buildDir,
        routes: { "/server_info": upstreamOrigin },
        runtimeServicesInfo,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") throw new Error("No port");
      const origin = `http://127.0.0.1:${address.port}`;

      const response = await getJson(`${origin}/server_info`);
      const body = response.body as {
        version?: string;
        runtime_services?: unknown;
      };

      expect(response.status).toBe(200);
      expect(body.version).toBe("1.48.0");
      expect(body.runtime_services).toEqual(JSON.parse(runtimeServicesInfo));
    });

    it("returns 502 when proxied /server_info target URL is invalid", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const runtimeServicesInfo = JSON.stringify({
        mode: "docker",
        services: {},
      });
      const server = await startStaticServer({
        port: 0,
        host: "127.0.0.1",
        dir: buildDir,
        routes: { "/server_info": "not-a-url" },
        runtimeServicesInfo,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") throw new Error("No port");
      const origin = `http://127.0.0.1:${address.port}`;

      const response = await getText(`${origin}/server_info`);

      expect(response.status).toBe(502);
      expect(response.body).toContain("Bad Gateway");
      expect(response.body).toContain("Invalid backend URL");
      expect(server.listening).toBe(true);
    });
  });

  describe("lock-to-cloud injection", () => {
    async function startServerLockedToCloud(dir: string, lockToCloud: string) {
      const server = await startStaticServer({
        port: 0,
        host: "127.0.0.1",
        dir,
        routes: {},
        lockToCloud,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") {
        throw new Error("Static server did not bind to a TCP port");
      }
      return `http://127.0.0.1:${address.port}`;
    }

    it("exposes the locked Cloud URL on window.__AGENT_CANVAS_LOCK_TO_CLOUD__", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServerLockedToCloud(
        buildDir,
        "https://cloud.example.com",
      );
      const body = await (await fetch(`${origin}/`)).text();

      expect(body).toContain("window.__AGENT_CANVAS_LOCK_TO_CLOUD__");
      expect(body).toContain('"https://cloud.example.com"');
    });

    it("injects lock-to-cloud into SPA fallback index.html", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServerLockedToCloud(
        buildDir,
        "https://cloud.example.com",
      );
      const response = await fetch(`${origin}/some/deep/route`);
      const body = await response.text();

      expect(response.status).toBe(200);
      expect(body).toContain("__AGENT_CANVAS_LOCK_TO_CLOUD__");
    });
  });

  describe("disable-telemetry injection", () => {
    async function startServerWithDisabledTelemetry(dir: string) {
      const origin = await startServer(dir, { disableTelemetry: true });
      return origin;
    }

    it("exposes window.__AGENT_CANVAS_DO_NOT_TRACK__ when enabled", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServerWithDisabledTelemetry(buildDir);
      const body = await (await fetch(`${origin}/`)).text();

      expect(body).toContain("window.__AGENT_CANVAS_DO_NOT_TRACK__=true");
    });

    it("injects the flag into the SPA fallback index.html", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServerWithDisabledTelemetry(buildDir);
      const response = await fetch(`${origin}/some/deep/route`);
      const body = await response.text();

      expect(response.status).toBe(200);
      expect(body).toContain("window.__AGENT_CANVAS_DO_NOT_TRACK__=true");
    });

    it("does not inject the flag when telemetry is not disabled", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServer(buildDir);
      const body = await (await fetch(`${origin}/`)).text();

      expect(body).not.toContain("__AGENT_CANVAS_DO_NOT_TRACK__");
    });
  });

  describe("session key injection", () => {
    async function startServerWithKey(dir: string, sessionApiKey: string) {
      const server = await startStaticServer({
        port: 0,
        host: "127.0.0.1",
        dir,
        routes: {},
        sessionApiKey,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") {
        throw new Error("Static server did not bind to a TCP port");
      }
      return `http://127.0.0.1:${address.port}`;
    }

    it("injects session key script into index.html", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServerWithKey(buildDir, "test-session-key");
      const response = await fetch(`${origin}/`);

      expect(response.status).toBe(200);
      const body = await response.text();
      expect(body).toContain("openhands-agent-server-config");
      expect(body).toContain("test-session-key");
      expect(body).toContain("sessionApiKey");
    });

    // Regression test: the published `agent-canvas` binary builds without
    // VITE_SESSION_API_KEY baked in, so the React app reads the key from
    // `window.__AGENT_CANVAS_SESSION_API_KEY__` (see
    // `getBakedSessionApiKey()` in `src/api/agent-server-config.ts`).
    // Without this assignment, `makeDefaultLocalBackend()` returns null
    // on a fresh install and the user gets the Manage Backends modal
    // instead of onboarding.
    it("exposes the session key on window.__AGENT_CANVAS_SESSION_API_KEY__", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServerWithKey(buildDir, "runtime-key");
      const response = await fetch(`${origin}/`);
      const body = await response.text();

      expect(body).toContain("window.__AGENT_CANVAS_SESSION_API_KEY__");
      expect(body).toContain('"runtime-key"');
      // The window assignment must precede the localStorage write so the
      // global is set even if storage access throws (private mode, etc.).
      const windowIdx = body.indexOf("__AGENT_CANVAS_SESSION_API_KEY__");
      const localStorageIdx = body.indexOf("openhands-agent-server-config");
      expect(windowIdx).toBeGreaterThan(-1);
      expect(localStorageIdx).toBeGreaterThan(-1);
      expect(windowIdx).toBeLessThan(localStorageIdx);
    });

    it("injects session key into SPA fallback index.html", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServerWithKey(buildDir, "fallback-key");
      const response = await fetch(`${origin}/some/deep/route`);

      expect(response.status).toBe(200);
      const body = await response.text();
      expect(body).toContain("fallback-key");
    });

    it("does not inject into non-html asset responses", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      mkdirSync(path.join(buildDir, "assets"));
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );
      writeFileSync(
        path.join(buildDir, "assets", "app.js"),
        "console.log('app');",
      );

      const origin = await startServerWithKey(buildDir, "should-not-inject");
      const response = await fetch(`${origin}/assets/app.js`);

      expect(response.status).toBe(200);
      const body = await response.text();
      expect(body).not.toContain("should-not-inject");
    });

    it("sets Cache-Control: no-store when session key is injected", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServerWithKey(buildDir, "cache-test-key");
      const response = await fetch(`${origin}/`);

      expect(response.headers.get("cache-control")).toBe("no-store");
    });

    it("sets Cache-Control: no-cache when session key is not present but other config is injected", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const server = await startStaticServer({
        port: 0,
        host: "127.0.0.1",
        dir: buildDir,
        routes: {},
        authRequired: true,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") throw new Error("No port");
      const origin = `http://127.0.0.1:${(address as { port: number }).port}`;

      const response = await fetch(`${origin}/`);

      expect(response.headers.get("cache-control")).toBe("no-cache");
    });

    it("does not inject the env session key when bound off-loopback", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      try {
        const server = await startStaticServer({
          ...parseArgs(["--host", "0.0.0.0"], {
            SESSION_API_KEY: "lan-secret",
          }),
          port: 0,
          dir: buildDir,
        });
        servers.push(server);
        const address = server.address();
        if (!address || typeof address === "string") {
          throw new Error("Static server did not bind to a TCP port");
        }
        const origin = `http://127.0.0.1:${address.port}`;
        const body = await (await fetch(`${origin}/`)).text();

        expect(body).not.toContain("lan-secret");
        expect(body).not.toContain("__AGENT_CANVAS_SESSION_API_KEY__");
        expect(body).toContain("__AGENT_CANVAS_AUTH_REQUIRED__");
        expect(warn).toHaveBeenCalled();
      } finally {
        warn.mockRestore();
      }
    });

    it("injects the session key off-loopback only with --allow-lan-session-key", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const server = await startStaticServer({
        port: 0,
        host: "0.0.0.0",
        dir: buildDir,
        routes: {},
        sessionApiKey: "container-key",
        allowLanSessionKey: true,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") {
        throw new Error("Static server did not bind to a TCP port");
      }
      const origin = `http://127.0.0.1:${address.port}`;
      const body = await (await fetch(`${origin}/`)).text();

      expect(body).toContain("container-key");
      expect(body).toContain("__AGENT_CANVAS_SESSION_API_KEY__");
    });

    it("escapes HTML special characters and script closing tags in injected values", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const payload = '</script><script>alert("xss")</script>';
      const origin = await startServerWithKey(buildDir, payload);
      const response = await fetch(`${origin}/`);
      const body = await response.text();

      expect(body).not.toContain("</script><script>");
      expect(body).toContain("\\u003c/script\\u003e\\u003cscript\\u003e");
    });

    it("does not inject when sessionApiKey is null", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const server = await startStaticServer({
        port: 0,
        host: "127.0.0.1",
        dir: buildDir,
        routes: {},
        sessionApiKey: null,
      });
      servers.push(server);
      const address = server.address();
      if (!address || typeof address === "string") throw new Error("No port");
      const origin = `http://127.0.0.1:${(address as { port: number }).port}`;

      const response = await fetch(`${origin}/`);
      const body = await response.text();
      expect(body).not.toContain("openhands-agent-server-config");
      expect(body).not.toContain("__AGENT_CANVAS_SESSION_API_KEY__");
    });

    it("injects session key into HTML without </head> tag (falls back to </body>)", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><body>no-head</body></html>",
      );

      const origin = await startServerWithKey(buildDir, "no-head-key");
      const response = await fetch(`${origin}/`);

      expect(response.status).toBe(200);
      const body = await response.text();
      expect(body).toContain("no-head-key");
      expect(body).toContain("openhands-agent-server-config");
      // Script should appear before </body>, not at the very front of the document
      expect(body.indexOf("no-head-key")).toBeLessThan(body.indexOf("</body>"));
      expect(body.indexOf("no-head-key")).toBeGreaterThan(0);
    });
  });

  it("serves nested build assets on all platforms", async () => {
    const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
    tempDirs.push(buildDir);
    mkdirSync(path.join(buildDir, "assets"));
    writeFileSync(path.join(buildDir, "index.html"), "<main>app</main>");
    writeFileSync(
      path.join(buildDir, "assets", "entry.client-test.js"),
      "export const loaded = true;\n",
    );

    const origin = await startServer(buildDir);
    const response = await fetch(`${origin}/assets/entry.client-test.js`);

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain(
      "application/javascript",
    );
    await expect(response.text()).resolves.toContain("loaded = true");
  });

  it("serves the current build after assets are replaced", async () => {
    const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
    tempDirs.push(buildDir);
    mkdirSync(path.join(buildDir, "assets"));
    writeFileSync(path.join(buildDir, "index.html"), "<main>app</main>");
    const oldAsset = path.join(buildDir, "assets", "old.js");
    writeFileSync(oldAsset, "old build");
    const origin = await startServer(buildDir);

    rmSync(oldAsset);
    writeFileSync(path.join(buildDir, "assets", "new.js"), "new build");

    const removed = await fetch(`${origin}/assets/old.js`);
    expect(removed.status).toBe(404);
    await removed.text();
    const replacement = await fetch(`${origin}/assets/new.js`);
    expect(replacement.status).toBe(200);
    expect(replacement.headers.get("cache-control")).toBe(
      "public, max-age=31536000, immutable",
    );
    await expect(replacement.text()).resolves.toBe("new build");
    const navigation = await fetch(`${origin}/conversations`, {
      headers: { Accept: "text/html" },
    });
    await expect(navigation.text()).resolves.toContain("<main>app</main>");
  });

  describe("base path mounting", () => {
    it("serves index.html and injects the base path under the mount", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );

      const origin = await startServer(buildDir, { basePath: "/canvas" });
      const response = await fetch(`${origin}/canvas/conversations/abc`);
      const body = await response.text();

      expect(response.status).toBe(200);
      expect(body).toContain("window.__AGENT_CANVAS_BASE_PATH__");
      expect(body).toContain('"/canvas"');
    });

    it("serves static assets from underneath the mount", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      mkdirSync(path.join(buildDir, "assets"));
      writeFileSync(path.join(buildDir, "index.html"), "<main>app</main>");
      writeFileSync(
        path.join(buildDir, "assets", "entry.client-test.js"),
        "export const loaded = true;\n",
      );

      const origin = await startServer(buildDir, { basePath: "/canvas" });
      const response = await fetch(
        `${origin}/canvas/assets/entry.client-test.js`,
      );

      expect(response.status).toBe(200);
      expect(response.headers.get("content-type")).toContain(
        "application/javascript",
      );
      await expect(response.text()).resolves.toContain("loaded = true");
    });

    it("redirects app routes outside the mount to the configured base path", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(path.join(buildDir, "index.html"), "<main>app</main>");

      const origin = await startServer(buildDir, { basePath: "/canvas" });
      const response = await fetch(`${origin}/conversations/abc?tab=files`, {
        redirect: "manual",
      });

      expect(response.status).toBe(308);
      expect(response.headers.get("location")).toBe(
        "/canvas/conversations/abc?tab=files",
      );
    });

    it("does not redirect asset-like requests outside the configured mount", async () => {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(path.join(buildDir, "index.html"), "<main>app</main>");

      const origin = await startServer(buildDir, { basePath: "/canvas" });
      const response = await fetch(`${origin}/assets/missing.js`, {
        redirect: "manual",
      });

      expect(response.status).toBe(404);
    });
  });

  describe("--no-referrer-prefix", () => {
    it("defaults to none", () => {
      expect(parseArgs([]).noReferrerPrefixes).toEqual([]);
    });

    it("collects repeated prefixes", () => {
      const config = parseArgs([
        "--no-referrer-prefix",
        "/vscode",
        "--no-referrer-prefix",
        "/editor",
      ]);
      expect(config.noReferrerPrefixes).toEqual(["/vscode", "/editor"]);
    });

    it("rejects a prefix without a leading slash", () => {
      expect(() => parseArgs(["--no-referrer-prefix", "vscode"])).toThrow(
        /must start with/,
      );
    });

    // Behaviour on a live proxied response is covered in ingress.test.ts,
    // which drives a real child process; an in-process proxy deadlocks against
    // the MSW interceptor this suite installs globally.
  });

  it("keeps paths confined to the static directory", async () => {
    const parentDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-parent-"));
    tempDirs.push(parentDir);
    const buildDir = path.join(parentDir, "build");
    mkdirSync(buildDir);
    writeFileSync(path.join(buildDir, "index.html"), "<main>app</main>");
    writeFileSync(path.join(parentDir, "secret.txt"), "secret\n");

    const origin = await startServer(buildDir);
    const response = await fetch(`${origin}/../secret.txt`);

    expect(response.status).not.toBe(200);
    await expect(response.text()).resolves.not.toContain("secret");
  });

  it("returns 502 when backend target URL is invalid", async () => {
    const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
    tempDirs.push(buildDir);
    writeFileSync(path.join(buildDir, "index.html"), "<main>app</main>");

    const server = await startStaticServer({
      port: 0,
      host: "127.0.0.1",
      dir: buildDir,
      routes: { "/api/invalid": "not-a-url" },
    });
    servers.push(server);
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("No port");
    const origin = `http://127.0.0.1:${address.port}`;

    const response = await getText(`${origin}/api/invalid/test`);

    expect(response.status).toBe(502);
    expect(response.body).toContain("Bad Gateway");
    expect(response.body).toContain("Invalid URL");
    expect(server.listening).toBe(true);
  });

  describe("proxied client identity (editor origin checks)", () => {
    // The agent-server editor proxy (software-agent-sdk,
    // openhands/agent_server/code_server_proxy.cjs) accepts a proxied
    // WebSocket upgrade only when the browser's Origin host matches the
    // client-facing host it is shown — `x-forwarded-host` or Host — and
    // code-server behind it re-checks the forwarded identity. On direct
    // access, httpxy rewrote Host to the local hop without setting
    // X-Forwarded-Host on upgrades (#19945). This fixture tests forwarding
    // against the editor proxy's Origin check, not cookie auth or code-server.
    type CapturedRequest = {
      url: string;
      headers: Record<string, string | string[] | undefined>;
    };
    const upstreamRequests: CapturedRequest[] = [];
    const upstreamSockets: Duplex[] = [];
    const clientSockets: Socket[] = [];

    const validOrigin = (req: IncomingMessage) => {
      if (!req.headers.origin) return true;
      try {
        const host = (
          (req.headers["x-forwarded-host"] as string | undefined) ||
          req.headers.host ||
          ""
        )
          .split(",")[0]
          .trim();
        return new URL(req.headers.origin as string).host === host;
      } catch {
        return false;
      }
    };

    const capture = (req: IncomingMessage) => {
      upstreamRequests.push({
        url: req.url ?? "",
        headers: { ...req.headers },
      });
    };

    // The editor's origin check applies on the editor route; the same
    // upstream doubles as a plain backend for other routes.
    const isEditorPath = (req: IncomingMessage) =>
      (req.url ?? "").startsWith("/vscode");

    function startEditorUpstream() {
      const upstream = createServer((req, res) => {
        capture(req);
        if (isEditorPath(req) && !validOrigin(req)) {
          res.writeHead(403).end();
          return;
        }
        res.writeHead(200, { "Content-Type": "text/plain" });
        res.end("editor");
      });
      upstream.on("upgrade", (req, socket) => {
        capture(req);
        if (isEditorPath(req) && !validOrigin(req)) {
          socket.end("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
          return;
        }
        upstreamSockets.push(socket);
        socket.write(
          "HTTP/1.1 101 Switching Protocols\r\n" +
            "Upgrade: websocket\r\n" +
            "Connection: Upgrade\r\n\r\n",
        );
      });
      return upstream;
    }

    function makeBuildDir() {
      const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
      tempDirs.push(buildDir);
      writeFileSync(
        path.join(buildDir, "index.html"),
        "<html><head></head><body>app</body></html>",
      );
      return buildDir;
    }

    async function startEditorProxyRoute(upstreamQuery = "") {
      const upstreamOrigin = await startHttpServer(startEditorUpstream());
      return startServer(makeBuildDir(), {
        routes: { "/vscode": `${upstreamOrigin}${upstreamQuery}` },
        vscodeBasePath: "/vscode",
      });
    }

    afterEach(() => {
      for (const socket of clientSockets.splice(0)) socket.destroy();
      for (const socket of upstreamSockets.splice(0)) socket.destroy();
      upstreamRequests.length = 0;
    });

    /**
     * Opens a raw WebSocket handshake against the static server, the way a
     * browser workbench does (Origin is always present), and resolves with
     * the response's status line once the upstream answers or closes.
     */
    function openUpgrade(
      origin: string,
      headers: Record<string, string>,
      path = "/vscode/?reconnectionToken=t&reconnection=false&skipWebSocketFrames=false",
    ): Promise<{ statusLine: string; buffered: string }> {
      const url = new URL(origin);
      return new Promise((resolve, reject) => {
        let buffered = "";
        let settled = false;
        const watchdog = setTimeout(() => {
          if (settled) return;
          settled = true;
          reject(new Error(`upgrade handshake timed out: ${buffered}`));
        }, 8000);
        const finish = (outcome: { statusLine: string; buffered: string }) => {
          if (settled) return;
          settled = true;
          clearTimeout(watchdog);
          resolve(outcome);
        };
        const socket = netConnect(
          { host: url.hostname, port: Number(url.port) },
          () => {
            const lines = [
              `GET ${path} HTTP/1.1`,
              `Host: ${headers.Host ?? url.host}`,
              "Upgrade: websocket",
              "Connection: Upgrade",
              "Sec-WebSocket-Version: 13",
              "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==",
            ];
            for (const [name, value] of Object.entries(headers)) {
              if (name !== "Host") lines.push(`${name}: ${value}`);
            }
            socket.write(`${lines.join("\r\n")}\r\n\r\n`);
          },
        );
        clientSockets.push(socket);
        socket.on("error", (err) => {
          if (settled) return;
          settled = true;
          clearTimeout(watchdog);
          reject(new Error(`${err}: ${buffered}`));
        });
        socket.on("data", (chunk: Buffer) => {
          buffered += chunk.toString("utf8");
          if (buffered.includes("\r\n")) {
            finish({
              statusLine: buffered.split("\r\n", 1)[0] ?? "",
              buffered,
            });
          }
        });
        socket.on("close", () => {
          finish({
            statusLine: buffered.split("\r\n", 1)[0] ?? "",
            buffered,
          });
        });
      });
    }

    it("forwards the client's Host and Origin on an editor WebSocket upgrade", async () => {
      const origin = await startEditorProxyRoute();
      const clientHost = new URL(origin).host;

      const { statusLine } = await openUpgrade(origin, {
        Origin: `http://${clientHost}`,
      });

      expect(statusLine).toBe("HTTP/1.1 101 Switching Protocols");
      const upgrade = upstreamRequests.at(-1);
      // The client-facing Host must reach the upstream — not the local hop
      // the request was forwarded to.
      expect(upgrade?.headers.host).toBe(clientHost);
      expect(upgrade?.headers.origin).toBe(`http://${clientHost}`);
      expect(upgrade?.headers["x-forwarded-host"]).toBe(clientHost);
      expect(upgrade?.headers["x-forwarded-port"]).toBe(new URL(origin).port);
      expect(upgrade?.headers["x-forwarded-proto"]).toBe("http");
    });

    it("keeps rejecting an upgrade whose Origin does not match the client-facing host", async () => {
      const origin = await startEditorProxyRoute();
      const clientHost = new URL(origin).host;
      const errorLog = vi.spyOn(console, "error").mockImplementation(() => {});

      try {
        const { buffered } = await openUpgrade(origin, {
          Origin: "http://evil.example",
        });

        // The editor proxy's origin check (kept intact upstream) answers
        // 401, and the rejection is passed back and logged.
        expect(buffered).toMatch(/^HTTP\/1\.1 401/);
        const upgrade = upstreamRequests.at(-1);
        expect(upgrade?.headers.host).toBe(clientHost);
        expect(upgrade?.headers.origin).toBe("http://evil.example");
        expect(errorLog.mock.calls).toEqual([
          ["[static:0:editor] WebSocket upstream rejected /vscode/: HTTP 401"],
        ]);
      } finally {
        errorLog.mockRestore();
      }
    });

    it("keeps the existing error label for editor proxy failures without an upstream response", async () => {
      const origin = await startServer(makeBuildDir(), {
        routes: { "/vscode": "not-a-url" },
        vscodeBasePath: "/vscode",
      });
      const errorLog = vi.spyOn(console, "error").mockImplementation(() => {});

      try {
        expect((await getText(`${origin}/vscode/`)).status).toBe(502);
        expect((await openUpgrade(origin, {}, "/vscode/")).statusLine).toBe("");
        expect(errorLog.mock.calls.map(([line]) => line)).toEqual([
          expect.stringMatching(/^\[static:0\] Proxy error for \/vscode\//),
          expect.stringMatching(
            /^\[static:0\] WebSocket proxy error for \/vscode\//,
          ),
        ]);
      } finally {
        errorLog.mockRestore();
      }
    });

    it.each([
      { host: "public.example", proto: "https", port: "443" },
      { host: "public.example", proto: "http", port: "80" },
      { host: "[2001:db8::2]:8443", proto: "https", port: "8443" },
    ])(
      "forwards $proto://$host with client-facing port $port instead of the local port",
      async ({ host, proto, port }) => {
        const origin = await startEditorProxyRoute();
        const { statusLine } = await openUpgrade(origin, {
          Host: host,
          Origin: `${proto}://${host}`,
          ...(proto === "https" ? { "X-Forwarded-Proto": proto } : {}),
        });

        expect(statusLine).toBe("HTTP/1.1 101 Switching Protocols");
        expect(upstreamRequests.at(-1)?.headers).toMatchObject({
          host,
          "x-forwarded-host": host,
          "x-forwarded-port": port,
          "x-forwarded-proto": proto,
        });
      },
    );

    it.each([
      {
        host: "public.example:9443",
        forwardedPort: "8443, 3001",
        expectedPort: "8443",
      },
      {
        host: "public.example:9443",
        forwardedPort: ", 3001",
        expectedPort: "9443",
      },
      {
        host: "public.example",
        forwardedPort: ", 3001",
        expectedPort: "443",
      },
    ])(
      "forwards port $expectedPort for Host $host and X-Forwarded-Port '$forwardedPort'",
      async ({ host, forwardedPort, expectedPort }) => {
        const origin = await startEditorProxyRoute();

        // Check the wire metadata independently of the upstream Origin check.
        const { statusLine } = await openUpgrade(origin, {
          Host: host,
          "X-Forwarded-Proto": "https",
          "X-Forwarded-Port": forwardedPort,
        });

        expect(statusLine).toBe("HTTP/1.1 101 Switching Protocols");
        expect(upstreamRequests.at(-1)?.headers["x-forwarded-port"]).toBe(
          expectedPort,
        );
      },
    );

    it("still proxies plain HTTP requests with the client-facing identity", async () => {
      const origin = await startEditorProxyRoute();
      const clientHost = new URL(origin).host;

      const response = await new Promise<{ status: number; body: string }>(
        (resolve, reject) => {
          const req = request(
            `${origin}/vscode/`,
            { headers: { Origin: `http://${clientHost}` } },
            (res) => {
              let body = "";
              res.setEncoding("utf8");
              res.on("data", (chunk) => {
                body += chunk;
              });
              res.on("end", () => {
                resolve({ status: res.statusCode ?? 0, body });
              });
            },
          );
          req.on("error", reject);
          req.end();
        },
      );

      expect(response.status).toBe(200);
      expect(response.body).toBe("editor");
      const captured = upstreamRequests.at(-1);
      expect(captured?.headers.host).toBe(clientHost);
      expect(captured?.headers.origin).toBe(`http://${clientHost}`);
      expect(captured?.headers["x-forwarded-host"]).toBe(clientHost);
      expect(captured?.headers["x-forwarded-port"]).toBe(new URL(origin).port);
      expect(captured?.headers["x-forwarded-proto"]).toBe("http");
      expect(captured?.headers["x-forwarded-for"]).toBe("127.0.0.1");
    });

    it.each([
      { scheme: "https", hostname: "localhost", serverName: "localhost" },
      { scheme: "wss", hostname: "localhost", serverName: "localhost" },
      { scheme: "https", hostname: "127.0.0.1", serverName: null },
      { scheme: "wss", hostname: "127.0.0.1", serverName: null },
    ])(
      "keys upstream TLS to $scheme://$hostname for HTTP and upgrades",
      async ({ scheme, hostname, serverName }) => {
        const serverNames: string[] = [];
        let failedHandshakes = 0;
        const upstream = createTlsServer({
          SNICallback(servername, done) {
            serverNames.push(servername);
            // No test certificate or trust override: stop after observing SNI.
            done(new Error("Test upstream stopped the handshake"));
          },
        });
        upstream.on("tlsClientError", () => {
          failedHandshakes += 1;
        });
        const address = new URL(await startHttpServer(upstream));
        const origin = await startServer(makeBuildDir(), {
          routes: { "/vscode": `${scheme}://${hostname}:${address.port}` },
          vscodeBasePath: "/vscode",
        });
        const host = "canvas.example";
        const errorLog = vi
          .spyOn(console, "error")
          .mockImplementation(() => {});

        try {
          const response = await new Promise<number>((resolve, reject) => {
            const req = request(
              `${origin}/vscode/`,
              { headers: { host } },
              (res) => {
                res.resume();
                res.on("end", () => resolve(res.statusCode ?? 0));
              },
            );
            req.on("error", reject);
            req.end();
          });
          expect(response).toBe(502);
          expect((await openUpgrade(origin, { Host: host })).statusLine).toBe(
            "",
          );
          expect(failedHandshakes).toBe(2);
          expect(serverNames).toEqual(
            serverName ? [serverName, serverName] : [],
          );
        } finally {
          errorLog.mockRestore();
        }
      },
    );

    it("accepts an editor upgrade without an Origin header (pinned upstream contract: no Origin means no check)", async () => {
      const origin = await startEditorProxyRoute();

      const { statusLine } = await openUpgrade(origin, {});

      expect(statusLine).toBe("HTTP/1.1 101 Switching Protocols");
      expect(upstreamRequests.at(-1)?.headers.host).toBe(new URL(origin).host);
    });

    it("still rejects an upgrade whose Origin is not a valid URL (pinned upstream contract)", async () => {
      const origin = await startEditorProxyRoute();

      const { buffered } = await openUpgrade(origin, {
        Origin: "not a url",
      });

      expect(buffered).toMatch(/^HTTP\/1\.1 401/);
    });

    it("uses the first forwarded host, port and scheme while preserving Host and the address chain", async () => {
      const origin = await startEditorProxyRoute();
      const errorLog = vi.spyOn(console, "error").mockImplementation(() => {});

      try {
        const { statusLine } = await openUpgrade(origin, {
          Host: "internal.example:3001",
          Origin: "https://public.example:8443",
          "X-Forwarded-Host": "public.example:8443, internal.example:3001",
          "X-Forwarded-Proto": "https, http",
          "X-Forwarded-Port": "8443, 3001",
          "X-Forwarded-For": "192.0.2.10",
        });

        expect(statusLine).toBe("HTTP/1.1 101 Switching Protocols");
        expect(upstreamRequests.at(-1)?.headers).toMatchObject({
          host: "internal.example:3001",
          origin: "https://public.example:8443",
          "x-forwarded-host": "public.example:8443",
          "x-forwarded-port": "8443",
          "x-forwarded-proto": "https",
          "x-forwarded-for": "192.0.2.10,127.0.0.1",
        });
        expect(errorLog).not.toHaveBeenCalled();
      } finally {
        errorLog.mockRestore();
      }
    });

    it.each([200, 403])(
      "returns HTTP %s using the forwarded authority when no forwarded port is set",
      async (status) => {
        const origin = await startEditorProxyRoute();
        const host = "internal.example:3001";
        const authority = "public.example:9443";
        const clientOrigin = `https://${status === 200 ? authority : host}`;
        const response = await getText(`${origin}/vscode/`, {
          Host: host,
          Origin: clientOrigin,
          "X-Forwarded-Host": authority,
          "X-Forwarded-Proto": "https",
        });

        expect(response.status).toBe(status);
        expect(response.body).toBe(status === 200 ? "editor" : "");
        expect(upstreamRequests.at(-1)?.headers).toMatchObject({
          host,
          origin: clientOrigin,
          "x-forwarded-host": authority,
          "x-forwarded-port": "9443",
          "x-forwarded-proto": "https",
        });
      },
    );

    it("leaves an empty first forwarded host for the upstream to reject", async () => {
      const origin = await startEditorProxyRoute();
      const forwardedHost = `, ${new URL(origin).host}`;
      const { statusLine } = await openUpgrade(origin, {
        Origin: origin,
        "X-Forwarded-Host": forwardedHost,
      });

      expect(statusLine).toMatch(/^HTTP\/1\.1 401/);
      expect(upstreamRequests.at(-1)?.headers["x-forwarded-host"]).toBe(
        forwardedHost,
      );
    });

    it("keeps one static diagnostics line counting both proxy policies", async () => {
      const upstream = await startHttpServer(startEditorUpstream());
      const origin = await startServer(makeBuildDir(), {
        routes: { "/proxy-test": upstream, "/vscode": upstream },
        vscodeBasePath: "/vscode",
      });
      expect((await getText(`${origin}/proxy-test`)).status).toBe(200);
      expect((await openUpgrade(origin, { Origin: origin })).statusLine).toBe(
        "HTTP/1.1 101 Switching Protocols",
      );

      const log = vi.spyOn(console, "log").mockImplementation(() => {});
      try {
        process.emit("SIGUSR1");
        const diagnostics = log.mock.calls
          .map(([line]) => String(line))
          .filter((line) => line.includes("active_http="));
        expect(diagnostics).toEqual([
          "[static:0] active_http=0 active_ws=1 total_http=1 total_ws=1 total_errors=0",
        ]);
      } finally {
        log.mockRestore();
      }
    });

    it("logs an upstream rejection without either credential-carrying query", async () => {
      const origin = await startEditorProxyRoute("?auth=TEST-UPSTREAM-TOKEN");
      const errorLog = vi.spyOn(console, "error").mockImplementation(() => {});

      try {
        // `tkn` carries the editor token (static-server's own
        // no-referrer-prefix comment): it must never reach the log line.
        const { statusLine } = await openUpgrade(
          origin,
          { Origin: "http://evil.example" },
          "/vscode/?tkn=SECRET-EDITOR-TOKEN",
        );
        expect(statusLine).toMatch(/^HTTP\/1\.1 401/);

        const logged = errorLog.mock.calls
          .map((call) => String(call[0]))
          .find((line) => line.includes("WebSocket upstream rejected"));
        expect(logged).toContain("/vscode/");
        expect(logged).not.toContain("SECRET-EDITOR-TOKEN");
        expect(logged).not.toContain("TEST-UPSTREAM-TOKEN");
        expect(logged).not.toContain("?");
      } finally {
        errorLog.mockRestore();
      }
    });

    it.each([
      { path: "/proxy-test/conversations", advertiseEditor: true },
      { path: "/vscode-extra/conversations", advertiseEditor: true },
      { path: "/vscode/override/conversations", advertiseEditor: true },
      { path: "/vscode/", advertiseEditor: false },
    ])(
      "keeps the previous proxy behavior on $path (editor advertised: $advertiseEditor)",
      async ({ path: requestPath, advertiseEditor }) => {
        // A neutral prefix: the test environment's request mocking intercepts
        // node http calls on the product's /api paths, which would never reach
        // this server.
        const upstreamOrigin = await startHttpServer(startEditorUpstream());
        const origin = await startServer(makeBuildDir(), {
          routes: {
            "/proxy-test": upstreamOrigin,
            "/vscode-extra": upstreamOrigin,
            // Even if the target is the same, a more specific route is not the
            // configured editor route and must keep its original proxy policy.
            "/vscode/override": upstreamOrigin,
            "/vscode": upstreamOrigin,
          },
          vscodeBasePath: advertiseEditor ? "/vscode" : null,
        });
        const upstreamHost = new URL(upstreamOrigin).host;

        const response = await getText(`${origin}${requestPath}`);
        expect(response.status).toBe(200);
        expect(response.body).toBe("editor");
        expect(upstreamRequests.at(-1)?.headers.host).toBe(upstreamHost);

        // A WebSocket upgrade on a non-editor route keeps the previous
        // behavior too: Host rewritten to the hop, and httpxy's own
        // x-forwarded-* pass (port from the static server's own port), not
        // the client-facing identity.
        const { statusLine } = await openUpgrade(origin, {}, requestPath);
        expect(statusLine).toBe("HTTP/1.1 101 Switching Protocols");
        const upgrade = upstreamRequests.at(-1);
        expect(upgrade?.headers.host).toBe(upstreamHost);
        expect(upgrade?.headers["x-forwarded-host"]).toBeUndefined();
        expect(upgrade?.headers["x-forwarded-port"]).toBe(new URL(origin).port);
      },
    );
  });
});

describe("static-server.mjs MARS web bridge (Docker arguments)", () => {
  const servers: Server[] = [];
  const tempDirs: string[] = [];

  afterEach(async () => {
    await Promise.all(
      servers.splice(0).map(
        (server) =>
          new Promise<void>((resolve) => {
            server.close(() => resolve());
          }),
      ),
    );
    for (const dir of tempDirs.splice(0)) {
      rmSync(dir, { recursive: true, force: true });
    }
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  function makeBuildDir() {
    const buildDir = mkdtempSync(path.join(tmpdir(), "agent-canvas-build-"));
    tempDirs.push(buildDir);
    writeFileSync(
      path.join(buildDir, "index.html"),
      "<html><head></head><body>app</body></html>",
    );
    return buildDir;
  }

  async function start(
    overrides: Partial<Parameters<typeof startStaticServer>[0]>,
  ) {
    const server = await startStaticServer({
      port: 0,
      dir: makeBuildDir(),
      routes: {},
      ...overrides,
    });
    servers.push(server);
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("Static server did not bind to a TCP port");
    }
    return `http://127.0.0.1:${address.port}`;
  }

  // docker/entrypoint.sh passes the key via SESSION_API_KEY with --host ::
  // but no --allow-lan-session-key: HTML omits the key while the opted-in
  // bridge still mounts behind it.
  it("mounts with MARS_WEB=1 and the entrypoint's non-loopback host and key", async () => {
    vi.stubEnv("MARS_WEB", "1");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const origin = await start({ host: "::", sessionApiKey: "docker-key" });
    expect(await (await fetch(`${origin}/mars/health`)).json()).toEqual({
      ok: true,
      authRequired: true,
    });
    const page = await (await fetch(`${origin}/`)).text();
    expect(page).not.toContain("docker-key");
    expect(
      warn.mock.calls.some((c) => String(c[0]).includes("[mars-web]")),
    ).toBe(false);
  });

  it("does not mount on a LAN bind with no key (public mode), and warns", async () => {
    vi.stubEnv("MARS_WEB", "1");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const origin = await start({ host: "::", authRequired: true });
    const response = await fetch(`${origin}/mars/health`);
    expect(response.headers.get("content-type")).toMatch(/text\/html/);
    expect(
      warn.mock.calls.some((c) =>
        /\[mars-web\].*not loopback/.test(String(c[0])),
      ),
    ).toBe(true);
  });

  // The public-mode instance binds `::` with no key, so a container-wide
  // MARS_WEB=1 would make it warn on every start, or, with MARS_WEB_KEY set
  // too, host a second bridge with its own sign-in state.
  it("pins the entrypoint's public-mode instance to MARS_WEB=0", () => {
    const entrypoint = readFileSync(
      path.resolve(import.meta.dirname, "../../docker/entrypoint.sh"),
      "utf-8",
    );
    expect(
      entrypoint.match(/^.*node \/opt\/agent-canvas\/static-server\.mjs/gm),
    ).toEqual([
      'SESSION_API_KEY="$EFFECTIVE_SESSION_KEY" node /opt/agent-canvas/static-server.mjs',
      "  MARS_WEB=0 node /opt/agent-canvas/static-server.mjs",
    ]);
  });

  it.each([undefined, "0"])(
    "does not mount with MARS_WEB=%s",
    async (value) => {
      vi.stubEnv("MARS_WEB", value);
      const origin = await start({ host: "127.0.0.1", sessionApiKey: "k" });
      const response = await fetch(`${origin}/mars/health`);
      expect(response.headers.get("content-type")).toMatch(/text\/html/);
    },
  );
});
