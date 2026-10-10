// @vitest-environment node
import http from "node:http";
import { once } from "node:events";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { http as mockHttp, passthrough } from "msw";
import { server as mockServer } from "#/mocks/node";
import { buildSafeDevConfigAsync } from "../../scripts/dev-safe.mjs";

describe("attached Agent Server preflight", () => {
  const servers: http.Server[] = [];
  const directories: string[] = [];

  afterEach(async () => {
    for (const server of servers.splice(0)) {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
    for (const directory of directories.splice(0))
      rmSync(directory, { recursive: true, force: true });
  });

  function makeEnv() {
    const directory = mkdtempSync(path.join(tmpdir(), "attached-server-"));
    directories.push(directory);
    return {
      OH_CANVAS_ATTACH_EXISTING_AGENT_SERVER: "1",
      OH_SESSION_API_KEY_PATH: path.join(directory, "api-key.txt"),
      OH_SECRET_KEY_PATH: path.join(directory, "secret-key.txt"),
    };
  }

  async function listen(
    options: { status?: number; info?: unknown; aliveStatus?: number } = {},
  ) {
    const requests: Array<{ method: string; path: string; key?: string }> = [];
    const server = http.createServer((req, res) => {
      requests.push({
        method: req.method!,
        path: req.url!,
        key: req.headers["x-session-api-key"] as string | undefined,
      });
      res.setHeader("Content-Type", "application/json");
      if (req.url === "/alive")
        res.writeHead(options.aliveStatus ?? 200).end('{"status":"ok"}');
      else if (req.url === "/api/settings")
        res.writeHead(options.status ?? 200).end("{}");
      else if (req.url === "/server_info")
        res.end(JSON.stringify(options.info ?? { version: "1.50.1" }));
      else res.writeHead(404).end("{}");
    });
    servers.push(server);
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const port = (server.address() as import("node:net").AddressInfo).port;
    mockServer.use(
      mockHttp.all(`http://127.0.0.1:${port}/*`, () => passthrough()),
    );
    return { port, requests };
  }

  // @spec LA-001 — Explicit attachment preserves server ownership
  it("reads an existing key without generating encryption state or requiring a free editor port", async () => {
    const env = makeEnv();
    writeFileSync(env.OH_SESSION_API_KEY_PATH, "persisted-test-key\n");
    const { port, requests } = await listen();
    const config = await buildSafeDevConfigAsync(process.cwd(), {
      ...env,
      OH_CANVAS_SAFE_BACKEND_PORT: String(port),
      OH_CANVAS_SAFE_VSCODE_PORT: String(port),
    });

    expect(config.sessionApiKey).toBe("persisted-test-key");
    expect(config.secretKey).toBeUndefined();
    expect(config.workingDir).toBeUndefined();
    expect(config.vscodeBasePath).toBeNull();
    expect(existsSync(env.OH_SECRET_KEY_PATH)).toBe(false);
    expect(readFileSync(env.OH_SESSION_API_KEY_PATH, "utf8")).toBe(
      "persisted-test-key\n",
    );
    expect(requests).toEqual(
      ["/alive", "/api/settings", "/server_info"].map((url) => ({
        method: "GET",
        path: url,
        key: "persisted-test-key",
      })),
    );
  });

  // @spec LA-002 — Verify an attached server before starting services
  it.each([
    { name: "wrong key", status: 401, message: /session key was rejected/ },
    {
      name: "unhealthy server",
      aliveStatus: 503,
      message: /verification failed/,
    },
    {
      name: "old version",
      info: { version: "1.0.0" },
      message: /valid version >=/,
    },
    {
      name: "unknown version",
      info: { version: "unknown" },
      message: /valid version >=/,
    },
    { name: "unrelated JSON service", info: {}, message: /valid version >=/ },
  ])("rejects $name", async ({ message, ...options }) => {
    const env = makeEnv();
    const { port } = await listen(options);
    await expect(
      buildSafeDevConfigAsync(process.cwd(), {
        ...env,
        LOCAL_BACKEND_API_KEY: "supplied-test-key",
        OH_CANVAS_SAFE_BACKEND_PORT: String(port),
      }),
    ).rejects.toThrow(message);
    expect(existsSync(env.OH_SESSION_API_KEY_PATH)).toBe(false);
    expect(existsSync(env.OH_SECRET_KEY_PATH)).toBe(false);
  });

  // @spec LA-001 — Explicit attachment preserves server ownership
  it("requires an existing session key rather than generating one", async () => {
    const env = makeEnv();
    await expect(buildSafeDevConfigAsync(process.cwd(), env)).rejects.toThrow(
      /existing Agent Server's session key/,
    );
    expect(existsSync(env.OH_SESSION_API_KEY_PATH)).toBe(false);
  });

  // @spec LA-001 — Explicit attachment preserves server ownership
  it("still rejects an occupied backend port without opt-in", async () => {
    const { port, requests } = await listen();
    await expect(
      buildSafeDevConfigAsync(process.cwd(), {
        ...makeEnv(),
        OH_CANVAS_ATTACH_EXISTING_AGENT_SERVER: "0",
        OH_CANVAS_SAFE_BACKEND_PORT: String(port),
      }),
    ).rejects.toThrow(/ports are already in use/);
    expect(requests).toEqual([]);
  });
});
