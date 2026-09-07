// @vitest-environment node
// These tests load `scripts/dev-safe.mjs` (via dev-extra-backend), which
// constructs file:// URLs relative to its own location via
// `new URL("../tools", import.meta.url)`. jsdom's URL constructor ignores
// file:// base URLs (it falls back to its document base, e.g.
// http://localhost:3000/), breaking that resolution; the Node environment
// has the standard WHATWG URL behavior that honors the file:// base.
import { spawn } from "node:child_process";
import { once } from "node:events";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import net from "node:net";
import { tmpdir } from "node:os";
import process from "node:process";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { buildExtraBackendConfig } from "../../scripts/dev-extra-backend.mjs";
import {
  buildSafeDevConfig,
  resetPersistedSessionApiKeyCache,
} from "../../scripts/dev-safe.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

describe("buildExtraBackendConfig", () => {
  const keyDirs: string[] = [];

  afterEach(() => {
    while (keyDirs.length > 0) {
      const dir = keyDirs.pop();
      if (dir) rmSync(dir, { recursive: true, force: true });
    }
    resetPersistedSessionApiKeyCache();
  });

  function isolatedKeyPath(): string {
    const dir = mkdtempSync(path.join(tmpdir(), "extra-backend-key-"));
    keyDirs.push(dir);
    return path.join(dir, "session-api-key.txt");
  }

  it("defaults to ports 18002/18003 distinct from the bundled instance", () => {
    const env = { OH_SESSION_API_KEY_PATH: isolatedKeyPath() };
    const bundled = buildSafeDevConfig(repoRoot, env);
    const extra = buildExtraBackendConfig(repoRoot, env);

    expect(extra.backendPort).toBe(18002);
    expect(extra.vscodePort).toBe(18003);
    expect(extra.backendBaseUrl).toBe("http://127.0.0.1:18002");
    expect(extra.backendHost).toBe("127.0.0.1:18002");
    expect(extra.backendPort).not.toBe(bundled.backendPort);
    expect(extra.vscodePort).not.toBe(bundled.vscodePort);
  });

  it("honors OH_CANVAS_EXTRA_BACKEND_PORT and OH_CANVAS_EXTRA_VSCODE_PORT", () => {
    const config = buildExtraBackendConfig(repoRoot, {
      OH_CANVAS_EXTRA_BACKEND_PORT: "29000",
      OH_CANVAS_EXTRA_VSCODE_PORT: "29001",
      OH_SESSION_API_KEY_PATH: isolatedKeyPath(),
    });

    expect(config.backendPort).toBe(29000);
    expect(config.vscodePort).toBe(29001);
    expect(config.backendBaseUrl).toBe("http://127.0.0.1:29000");
  });

  it("shares state dir, conversations, bash events, and secret key with the bundled config", () => {
    const env = {
      OH_CANVAS_SAFE_STATE_DIR: "/tmp/canvas-state",
      OH_SESSION_API_KEY_PATH: isolatedKeyPath(),
    };
    const bundled = buildSafeDevConfig(repoRoot, env);
    const extra = buildExtraBackendConfig(repoRoot, env);

    expect(extra.stateDir).toBe(bundled.stateDir);
    expect(extra.conversationsPath).toBe(bundled.conversationsPath);
    expect(extra.bashEventsDir).toBe(bundled.bashEventsDir);
    expect(extra.tmuxTmpDir).toBe(bundled.tmuxTmpDir);
    expect(extra.secretKey).toBe(bundled.secretKey);
  });

  it("rejects an invalid OH_CANVAS_EXTRA_BACKEND_PORT", () => {
    expect(() =>
      buildExtraBackendConfig(repoRoot, {
        OH_CANVAS_EXTRA_BACKEND_PORT: "not-a-port",
        OH_SESSION_API_KEY_PATH: isolatedKeyPath(),
      }),
    ).toThrow(/Invalid port/);
  });
});

describe("dev-extra-backend CLI shutdown", () => {
  it.skipIf(process.platform === "win32")(
    "cleans up the detached agent-server when the launcher receives SIGHUP",
    async () => {
      // The agent-server is spawned detached (getProcessTreeSpawnOptions), so
      // killing the launcher does not kill it. Drive the real launcher with a
      // stub agent-server, then SIGHUP the launcher and assert the stub's port
      // is released rather than held by a survivor.
      const stubDir = mkdtempSync(path.join(tmpdir(), "dev-extra-sighup-"));
      const stubJs = path.join(stubDir, "stub-agent-server.mjs");
      const uvxStub = path.join(stubDir, "uvx");

      writeFileSync(
        stubJs,
        [
          'import net from "node:net";',
          'const portIndex = process.argv.indexOf("--port");',
          "const port = Number(process.argv[portIndex + 1]);",
          "const server = net.createServer(() => {});",
          'server.listen(port, "127.0.0.1", () => {',
          '  console.log("STUB_LISTENING", process.pid, port);',
          "});",
          "setInterval(() => {}, 1_000);",
        ].join("\n"),
      );
      writeFileSync(
        uvxStub,
        `#!/bin/sh\nexec ${JSON.stringify(process.execPath)} ${JSON.stringify(stubJs)} "$@"\n`,
      );
      chmodSync(uvxStub, 0o755);

      const backendPort = await new Promise<number>((resolve, reject) => {
        const probe = net.createServer();
        probe.on("error", reject);
        probe.listen(0, "127.0.0.1", () => {
          const address = probe.address();
          const port =
            typeof address === "object" && address ? address.port : 0;
          probe.close(() => resolve(port));
        });
      });

      const isPortListening = async () =>
        new Promise<boolean>((resolve) => {
          const socket = net
            .connect(backendPort, "127.0.0.1")
            .on("connect", () => {
              socket.destroy();
              resolve(true);
            })
            .on("error", () => resolve(false));
        });

      const launcher = spawn(
        process.execPath,
        ["scripts/dev-extra-backend.mjs"],
        {
          cwd: repoRoot,
          env: {
            ...process.env,
            PATH: `${stubDir}${path.delimiter}${process.env.PATH ?? ""}`,
            OH_CANVAS_EXTRA_BACKEND_PORT: String(backendPort),
          },
          stdio: ["ignore", "pipe", "pipe"],
        },
      );

      let output = "";
      let stubPid: number | undefined;
      const capture = (chunk: Buffer) => {
        output += chunk.toString();
        const stubMatch = output.match(/STUB_LISTENING (\d+)/);
        if (stubMatch) stubPid = Number(stubMatch[1]);
      };
      launcher.stdout.on("data", capture);
      launcher.stderr.on("data", capture);

      try {
        const readyDeadline = Date.now() + 20_000;
        let listening = false;
        while (!listening && Date.now() < readyDeadline) {
          if (launcher.exitCode !== null) break;
          listening = await isPortListening();
          if (!listening) await delay(100);
        }
        expect(listening, output).toBe(true);

        launcher.kill("SIGHUP");
        await Promise.race([once(launcher, "exit"), delay(10_000)]);

        // shutdown() forwards SIGTERM, then SIGKILLs after 3s.
        const freeDeadline = Date.now() + 12_000;
        let stillListening = true;
        while (stillListening && Date.now() < freeDeadline) {
          stillListening = await isPortListening();
          if (stillListening) await delay(200);
        }
        expect(stillListening, output).toBe(false);
      } finally {
        if (launcher.exitCode === null) launcher.kill("SIGKILL");
        // The stub is a detached process-group leader, so killing the launcher
        // does not reap it. Without this, the regression path this test exists
        // to catch would itself leave the stub holding its port indefinitely.
        if (stubPid !== undefined) {
          try {
            process.kill(-stubPid, "SIGKILL");
          } catch {
            // Already gone, which is the passing path.
          }
        }
        rmSync(stubDir, { recursive: true, force: true });
      }
    },
    45_000,
  );
});
