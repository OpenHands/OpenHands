// @vitest-environment node
// These tests load `scripts/dev-with-automation.mjs` and `scripts/dev-safe.mjs`,
// which construct file:// URLs relative to their own location via
// `new URL("../tools", import.meta.url)`. jsdom's URL constructor ignores
// file:// base URLs (it falls back to its document base, e.g.
// http://localhost:3000/), breaking that resolution; the Node environment
// has the standard WHATWG URL behavior that honors the file:// base.
import net from "node:net";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it, afterEach } from "vitest";
import {
  buildAgentServerAutomationEnv,
  buildAutomationCommand,
  buildAutomationTelemetryEnv,
  buildConfig,
  buildRouteArgs,
  buildViteBackendEnv,
  getFrontendBackend,
  getLocalServiceRoutes,
  setServiceLogListener,
  spawnService,
  DEFAULT_AUTOMATION_REPO,
  DEFAULT_AUTOMATION_PACKAGE,
  DEFAULT_AUTOMATION_VERSION,
  DEFAULT_BACKEND_PORT,
  DEFAULT_AUTOMATION_PORT,
} from "../../scripts/dev-with-automation.mjs";
import { resetPersistedSessionApiKeyCache } from "../../scripts/dev-safe.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

describe("buildAutomationCommand", () => {
  it("uses released PyPI version by default", () => {
    const cmd = buildAutomationCommand({});

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toContain("--from");
    expect(cmd.args).toContain(
      `${DEFAULT_AUTOMATION_PACKAGE}==${DEFAULT_AUTOMATION_VERSION}`,
    );
    expect(cmd.args).toContain("uvicorn");
    expect(cmd.args).toContain("openhands.automation.app:app");
    expect(cmd.source).toBe(`PyPI (${DEFAULT_AUTOMATION_VERSION}, default)`);
  });

  it("uses custom git ref from OH_AUTOMATION_GIT_REF", () => {
    const cmd = buildAutomationCommand({
      OH_AUTOMATION_GIT_REF: "feat/my-feature",
    });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toContain("--from");
    expect(cmd.args).toContain(
      `git+${DEFAULT_AUTOMATION_REPO}@feat/my-feature`,
    );
    expect(cmd.source).toBe("git (feat/my-feature)");
  });

  it("uses custom repo with git ref", () => {
    const cmd = buildAutomationCommand({
      OH_AUTOMATION_REPO: "https://github.com/MyOrg/my-automation",
      OH_AUTOMATION_GIT_REF: "main",
    });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toContain(
      "git+https://github.com/MyOrg/my-automation@main",
    );
  });

  it("uses both custom repo and ref together", () => {
    const cmd = buildAutomationCommand({
      OH_AUTOMATION_REPO: "https://github.com/MyOrg/my-automation",
      OH_AUTOMATION_GIT_REF: "v1.0.0",
    });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toContain(
      "git+https://github.com/MyOrg/my-automation@v1.0.0",
    );
    expect(cmd.source).toBe("git (v1.0.0)");
  });

  it("supports commit SHA as git ref", () => {
    const cmd = buildAutomationCommand({
      OH_AUTOMATION_GIT_REF: "abc123def456",
    });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toContain(`git+${DEFAULT_AUTOMATION_REPO}@abc123def456`);
    expect(cmd.source).toBe("git (abc123def456)");
  });

  it("uses specific PyPI version when OH_AUTOMATION_VERSION is set", () => {
    const cmd = buildAutomationCommand({
      OH_AUTOMATION_VERSION: "1.0.0",
    });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toContain(`${DEFAULT_AUTOMATION_PACKAGE}==1.0.0`);
    expect(cmd.source).toBe("PyPI (1.0.0)");
  });

  it("git ref takes precedence over version", () => {
    const cmd = buildAutomationCommand({
      OH_AUTOMATION_GIT_REF: "main",
      OH_AUTOMATION_VERSION: "1.0.0",
    });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toContain(`git+${DEFAULT_AUTOMATION_REPO}@main`);
    expect(cmd.args).not.toContain(`${DEFAULT_AUTOMATION_PACKAGE}==1.0.0`);
    expect(cmd.source).toBe("git (main)");
  });
});

describe("buildAgentServerAutomationEnv", () => {
  it("exposes the session API key as OPENHANDS_AUTOMATION_API_KEY for agent curl commands", () => {
    expect(
      buildAgentServerAutomationEnv({ sessionApiKey: "shared-session-key" }),
    ).toEqual({
      OPENHANDS_AUTOMATION_API_KEY: "shared-session-key",
    });
  });
});

describe("buildAutomationTelemetryEnv", () => {
  it("provides the shared frontend PostHog key to automation by default", () => {
    expect(buildAutomationTelemetryEnv({})).toEqual({
      AUTOMATION_POSTHOG_API_KEY:
        "phc_kBtz5nKmxVRRQ7HtPwr2QX9eMC5j65zE86QKocVNwb4U",
      AUTOMATION_POSTHOG_HOST: "https://us.i.posthog.com",
    });
  });

  it("prefers explicit automation telemetry settings", () => {
    expect(
      buildAutomationTelemetryEnv({
        AUTOMATION_POSTHOG_API_KEY: "phc_auto",
        AUTOMATION_POSTHOG_HOST: "https://auto.example",
        VITE_POSTHOG_API_KEY: "phc_frontend",
        VITE_POSTHOG_HOST: "https://frontend.example",
      }),
    ).toEqual({
      AUTOMATION_POSTHOG_API_KEY: "phc_auto",
      AUTOMATION_POSTHOG_HOST: "https://auto.example",
    });
  });

  it("uses explicit frontend telemetry settings when automation settings are absent", () => {
    expect(
      buildAutomationTelemetryEnv({
        VITE_POSTHOG_API_KEY: "phc_frontend",
        VITE_POSTHOG_HOST: "https://frontend.example",
      }),
    ).toEqual({
      AUTOMATION_POSTHOG_API_KEY: "phc_frontend",
      AUTOMATION_POSTHOG_HOST: "https://frontend.example",
    });
  });

  it("does not default automation telemetry when do-not-track is enabled", () => {
    expect(buildAutomationTelemetryEnv({ VITE_DO_NOT_TRACK: "1" })).toEqual({});
  });
});

describe("buildConfig", () => {
  const servers: net.Server[] = [];
  const keyDirs: string[] = [];

  afterEach(() => {
    for (const server of servers) {
      server.close();
    }
    servers.length = 0;
    while (keyDirs.length > 0) {
      const dir = keyDirs.pop();
      if (dir) rmSync(dir, { recursive: true, force: true });
    }
    resetPersistedSessionApiKeyCache();
  });

  /**
   * Build an env that points persisted dev API key files at a fresh temp dir,
   * so tests don't write to the user's real ~/.openhands/agent-canvas files.
   *
   * Also redirects all service ports to high port numbers so that buildConfig's
   * assertPortsFree check passes even when a real dev stack is running on the
   * default ports (18000, 18001, 3001, 8000).
   */
  function envWithIsolatedKeyPath(
    extra: Record<string, string> = {},
  ): Record<string, string> {
    const dir = mkdtempSync(path.join(tmpdir(), "buildconfig-key-"));
    keyDirs.push(dir);
    return {
      OH_SESSION_API_KEY_PATH: path.join(dir, "session-api-key.txt"),
      // High ports that are almost certainly free, so assertPortsFree passes.
      PORT: "19902",
      OH_CANVAS_SAFE_BACKEND_PORT: "19900",
      OH_CANVAS_SAFE_AUTOMATION_PORT: "19901",
      OH_CANVAS_SAFE_VITE_PORT: "19903",
      ...extra,
    };
  }

  it("builds default config with correct ports", async () => {
    const config = await buildConfig({}, envWithIsolatedKeyPath());

    // Ports should be allocated (either defaults if free, or alternatives)
    expect(typeof config.ingressPort).toBe("number");
    expect(config.ingressPort).toBeGreaterThan(0);
    expect(typeof config.agentServerPort).toBe("number");
    expect(config.agentServerPort).toBeGreaterThan(0);
    expect(typeof config.autoBackendPort).toBe("number");
    expect(config.autoBackendPort).toBeGreaterThan(0);
    expect(typeof config.vitePort).toBe("number");
    expect(config.vitePort).toBeGreaterThan(0);
    expect(config.vscodePort).toBe(config.agentServerPort + 1000);

    // All four main ports should be unique
    const ports = new Set([
      config.ingressPort,
      config.agentServerPort,
      config.autoBackendPort,
      config.vitePort,
    ]);
    expect(ports.size).toBe(4);
  });

  it("respects preferred port from args when available", async () => {
    // Use a high port unlikely to be busy
    const preferredPort = 19500;
    const config = await buildConfig(
      { port: preferredPort },
      envWithIsolatedKeyPath(),
    );

    expect(config.ingressPort).toBe(preferredPort);
  });

  it("throws when ingress port is busy", async () => {
    const busyPort = 8100;

    // Block port 8100
    const server = net.createServer();
    await new Promise<void>((resolve, reject) => {
      server.listen(busyPort, "127.0.0.1", () => {
        servers.push(server);
        resolve();
      });
      server.on("error", reject);
    });

    // Should throw instead of falling back to a different port
    await expect(
      buildConfig({ port: busyPort }, envWithIsolatedKeyPath()),
    ).rejects.toThrow(/ingress.*port 8100/i);
  });

  it("allocates valid ports for all services", async () => {
    const config = await buildConfig({}, envWithIsolatedKeyPath());

    // All service ports should be valid
    expect(config.agentServerPort).toBeGreaterThan(0);
    expect(config.autoBackendPort).toBeGreaterThan(0);
    expect(config.vitePort).toBeGreaterThan(0);
    expect(config.vscodePort).toBeGreaterThan(0);

    // All service ports should be different from each other
    const servicePorts = [
      config.agentServerPort,
      config.autoBackendPort,
      config.vitePort,
      config.ingressPort,
    ];
    expect(new Set(servicePorts).size).toBe(servicePorts.length);
  });

  it("respects preferred PORT from env when available", async () => {
    // Use a high port unlikely to be busy
    const preferredPort = "19501";
    const config = await buildConfig(
      {},
      envWithIsolatedKeyPath({ PORT: preferredPort }),
    );

    expect(config.ingressPort).toBe(19501);
  });

  it("args.port takes precedence over env.PORT", async () => {
    // Use high ports unlikely to be busy
    const config = await buildConfig(
      { port: 19502 },
      envWithIsolatedKeyPath({ PORT: "19599" }),
    );

    expect(config.ingressPort).toBe(19502);
  });

  it("applies automationGitRef from args to env", async () => {
    const env = envWithIsolatedKeyPath();
    await buildConfig({ automationGitRef: "my-branch" }, env);

    expect(env.OH_AUTOMATION_GIT_REF).toBe("my-branch");
  });

  it("applies automationRepo from args to env", async () => {
    const env = envWithIsolatedKeyPath();
    await buildConfig({ automationRepo: "https://example.com/repo" }, env);

    expect(env.OH_AUTOMATION_REPO).toBe("https://example.com/repo");
  });

  it("uses correct state directory path", async () => {
    const config = await buildConfig({}, envWithIsolatedKeyPath());

    expect(config.stateDir).toBe(
      path.join(homedir(), ".openhands", "agent-canvas"),
    );
  });

  it("passes verbose flag through", async () => {
    const config = await buildConfig(
      { verbose: true },
      envWithIsolatedKeyPath(),
    );

    expect(config.verbose).toBe(true);
  });

  it("sessionApiKey is a 64-char hex string by default", async () => {
    const config = await buildConfig({}, envWithIsolatedKeyPath());

    expect(config.sessionApiKey).toMatch(/^[0-9a-f]{64}$/);
  });

  it("falls back to a freshly persisted session API key by default", async () => {
    const config = await buildConfig({}, envWithIsolatedKeyPath());

    // Default is a 64-char hex string (256-bit random key) read from /
    // written to OH_SESSION_API_KEY_PATH.
    expect(config.sessionApiKey).toMatch(/^[0-9a-f]{64}$/);
  });

  it("reuses the persisted session API key across calls (stable across restarts)", async () => {
    const env = envWithIsolatedKeyPath();
    const first = await buildConfig({}, env);

    // Simulate a fresh process invocation (the file on disk should be
    // what makes the key stable).
    resetPersistedSessionApiKeyCache();

    const second = await buildConfig({}, env);

    expect(second.sessionApiKey).toBe(first.sessionApiKey);
  });

  it("reads sessionApiKey from LOCAL_BACKEND_API_KEY", async () => {
    const config = await buildConfig(
      {},
      { ...envWithIsolatedKeyPath(), LOCAL_BACKEND_API_KEY: "my-api-key" },
    );

    expect(config.sessionApiKey).toBe("my-api-key");
  });
});

describe("stack mode routing", () => {
  const keyDirs: string[] = [];

  afterEach(() => {
    while (keyDirs.length > 0) {
      const dir = keyDirs.pop();
      if (dir) rmSync(dir, { recursive: true, force: true });
    }
    resetPersistedSessionApiKeyCache();
  });

  function envWithIsolatedKeyPath(
    extra: Record<string, string> = {},
  ): Record<string, string> {
    const dir = mkdtempSync(path.join(tmpdir(), "stack-mode-key-"));
    keyDirs.push(dir);
    return {
      OH_SESSION_API_KEY_PATH: path.join(dir, "session-api-key.txt"),
      PORT: "19802",
      OH_CANVAS_SAFE_BACKEND_PORT: "19800",
      OH_CANVAS_SAFE_AUTOMATION_PORT: "19801",
      OH_CANVAS_SAFE_VITE_PORT: "19803",
      ...extra,
    };
  }

  it("uses only a frontend default route in frontend-only mode", async () => {
    const config = await buildConfig(
      { frontendOnly: true },
      envWithIsolatedKeyPath(),
    );

    expect(config.launchFrontend).toBe(true);
    expect(config.launchAgentServer).toBe(false);
    expect(config.launchAutomation).toBe(false);
    expect(getLocalServiceRoutes(config)).toEqual([]);
    expect(getFrontendBackend(config)).toBe(
      `http://localhost:${config.vitePort}`,
    );
    expect(buildRouteArgs(getLocalServiceRoutes(config))).toEqual([]);
  });

  it("does not bake a host workspace path in frontend-only mode by default", async () => {
    const config = await buildConfig(
      { frontendOnly: true },
      envWithIsolatedKeyPath(),
    );

    expect(config.viteWorkingDir).toBeUndefined();
  });

  it("honors explicit frontend-only VITE_WORKING_DIR values", async () => {
    const config = await buildConfig(
      { frontendOnly: true },
      envWithIsolatedKeyPath({ VITE_WORKING_DIR: "workspace/project" }),
    );

    expect(config.viteWorkingDir).toBe("workspace/project");
  });

  it("bakes the host workspace path when this launcher starts the agent-server", async () => {
    const config = await buildConfig({}, envWithIsolatedKeyPath());

    expect(config.viteWorkingDir).toBe(
      path.join(config.stateDir, "workspaces"),
    );
  });

  it("points frontend-only Vite at a separately running backend by default", async () => {
    const config = await buildConfig(
      { frontendOnly: true },
      envWithIsolatedKeyPath(),
    );

    expect(buildViteBackendEnv(config, {})).toEqual({
      VITE_BACKEND_HOST: "127.0.0.1:8000",
      VITE_BACKEND_BASE_URL: "http://127.0.0.1:8000",
    });
  });

  it("keeps full-stack Vite pointed at this launcher's ingress", async () => {
    const config = await buildConfig({}, envWithIsolatedKeyPath());

    expect(buildViteBackendEnv(config, {})).toEqual({
      VITE_BACKEND_HOST: `127.0.0.1:${config.ingressPort}`,
      VITE_BACKEND_BASE_URL: `http://127.0.0.1:${config.ingressPort}`,
    });
  });

  it("allows frontend-only Vite to target an explicit backend URL", async () => {
    const config = await buildConfig(
      { frontendOnly: true },
      envWithIsolatedKeyPath(),
    );

    expect(
      buildViteBackendEnv(config, {
        VITE_BACKEND_BASE_URL: "https://backend.example.test",
      }),
    ).toEqual({
      VITE_BACKEND_HOST: "backend.example.test",
      VITE_BACKEND_BASE_URL: "https://backend.example.test",
    });
  });

  it("routes only local services through IPv4 in backend-only mode", async () => {
    const config = await buildConfig(
      { backendOnly: true },
      envWithIsolatedKeyPath(),
    );

    expect(config.launchFrontend).toBe(false);
    expect(config.launchAgentServer).toBe(true);
    expect(config.launchAutomation).toBe(true);
    expect(getFrontendBackend(config)).toBeNull();

    const routes = getLocalServiceRoutes(config);
    expect(routes).toContainEqual([
      "/api/automation",
      `http://127.0.0.1:${config.autoBackendPort}`,
    ]);
    expect(routes).toContainEqual([
      "/api",
      `http://127.0.0.1:${config.agentServerPort}`,
    ]);

    const routeArgs = buildRouteArgs(routes);
    expect(routeArgs).toContain(
      `/api/automation=http://127.0.0.1:${config.autoBackendPort}`,
    );
    expect(routeArgs).toContain(
      `/server_info=http://127.0.0.1:${config.agentServerPort}`,
    );
    expect(routeArgs).not.toContain("--default");
  });

  it("rejects mutually exclusive partial-stack modes", async () => {
    await expect(
      buildConfig(
        { frontendOnly: true, backendOnly: true },
        envWithIsolatedKeyPath(),
      ),
    ).rejects.toThrow(/cannot be used together/);
  });
});

describe("default constants", () => {
  it("has expected default automation repo", () => {
    expect(DEFAULT_AUTOMATION_REPO).toBe(
      "https://github.com/OpenHands/automation",
    );
  });

  it("has expected default automation package", () => {
    expect(DEFAULT_AUTOMATION_PACKAGE).toBe("openhands-automation");
  });

  it("has expected default backend port", () => {
    expect(DEFAULT_BACKEND_PORT).toBe(18000);
  });

  it("has expected default automation port", () => {
    expect(DEFAULT_AUTOMATION_PORT).toBe(18001);
  });
});

describe("setServiceLogListener", () => {
  afterEach(() => {
    // Always clear the listener so it doesn't leak between tests.
    setServiceLogListener(null);
  });

  it("forwards stdout, stderr, and exit lines to the listener", async () => {
    const captured: Array<{ name: string; line: string; level: string }> = [];
    setServiceLogListener((name: string, line: string, level: string) => {
      captured.push({ name, line, level });
    });

    // Spawn a tiny inline script that writes to stdout, stderr, and exits
    // non-zero so we exercise all three log paths. `process.execPath` is
    // the same Node binary running the test so this is portable.
    const proc = spawnService(
      "log-listener-test",
      process.execPath,
      [
        "-e",
        // Use double-quoted JS to avoid shell quoting differences.
        'process.stdout.write("hello-stdout\\n"); process.stderr.write("hello-stderr\\n"); process.exit(2);',
      ],
      {},
    );

    await once(proc, "exit");
    // Give the stdout/stderr 'data' handlers and the exit handler a tick to
    // run after the process has actually exited.
    await delay(50);

    const lines = captured.map((c) => `${c.level}:${c.line}`);
    expect(lines).toContain("stdout:hello-stdout");
    expect(lines).toContain("stderr:hello-stderr");
    expect(lines).toContain("error:exited with code 2");

    // Every captured line must be tagged with our service name.
    expect(captured.every((c) => c.name === "log-listener-test")).toBe(true);
  });

  it("is a no-op once the listener is cleared", async () => {
    const captured: string[] = [];
    setServiceLogListener((_n: string, line: string) => captured.push(line));
    setServiceLogListener(null);

    const proc = spawnService(
      "log-listener-clear-test",
      process.execPath,
      ["-e", 'process.stdout.write("should-not-be-seen\\n");'],
      {},
    );

    await once(proc, "exit");
    await delay(50);

    expect(captured).toHaveLength(0);
  });

  it("ignores non-function values without throwing", () => {
    // Passing anything that isn't a function (undefined, null, a string) must
    // be safe — main() relies on this so it can always call
    // setServiceLogListener(onServiceLog) regardless of whether the embedder
    // supplied a callback.
    expect(() => setServiceLogListener(undefined)).not.toThrow();
    expect(() => setServiceLogListener(null)).not.toThrow();
    expect(() => setServiceLogListener("not a function")).not.toThrow();
  });

  it("swallows errors thrown by the listener so a buggy embedder cannot kill the dev stack", async () => {
    // A listener that throws on every call must not propagate; the stdout
    // handler in spawnService is fire-and-forget. We can't easily observe
    // an unhandled rejection from within Vitest, but if we can run a process
    // to completion without the test failing, the catch is doing its job.
    setServiceLogListener(() => {
      throw new Error("buggy listener");
    });

    const proc = spawnService(
      "log-listener-throws-test",
      process.execPath,
      ["-e", 'process.stdout.write("line\\n"); process.exit(0);'],
      {},
    );

    const [code] = await once(proc, "exit");
    expect(code).toBe(0);
  });
});

describe("dev-with-automation CLI", () => {
  it.skipIf(process.platform === "win32")(
    "cleans up detached services when the launcher receives SIGHUP",
    async () => {
      const moduleUrl = pathToFileURL(
        path.join(repoRoot, "scripts", "dev-with-automation.mjs"),
      ).href;
      const fixtureSource = [
        'import net from "node:net";',
        "const server = net.createServer(() => {});",
        'server.listen(0, "127.0.0.1", () => console.log("READY", process.pid, server.address().port));',
      ].join("\n");
      const supervisorSource = [
        `import { spawnService } from ${JSON.stringify(moduleUrl)};`,
        `const fixture = spawnService("fixture", process.execPath, ["--input-type=module", "--eval", ${JSON.stringify(fixtureSource)}]);`,
        'console.log("SPAWNED", fixture.pid);',
        "setInterval(() => {}, 1_000);",
      ].join("\n");

      const supervisor = spawn(
        process.execPath,
        ["--input-type=module", "--eval", supervisorSource],
        {
          cwd: repoRoot,
          stdio: ["ignore", "pipe", "pipe"],
        },
      );
      let output = "";
      let fixturePid: number | undefined;
      let fixturePort: number | undefined;
      const capture = (chunk: Buffer) => {
        output += chunk.toString();
        const spawnedMatch = output.match(/SPAWNED (\d+)/);
        const readyMatch = output.match(/READY \d+ (\d+)/);
        if (spawnedMatch) fixturePid = Number(spawnedMatch[1]);
        if (readyMatch) fixturePort = Number(readyMatch[1]);
      };
      supervisor.stdout.on("data", capture);
      supervisor.stderr.on("data", capture);

      try {
        const readyDeadline = Date.now() + 5_000;
        while (!fixturePort && Date.now() < readyDeadline) {
          if (supervisor.exitCode !== null) break;
          await delay(25);
        }
        expect(fixturePid, output).toBeDefined();
        expect(fixturePort, output).toBeDefined();

        supervisor.kill("SIGHUP");
        const exitResult = await Promise.race([
          once(supervisor, "exit").then(([code, signal]) => ({
            code,
            signal,
            timedOut: false,
          })),
          delay(6_000).then(() => ({
            code: null,
            signal: null,
            timedOut: true,
          })),
        ]);

        expect(exitResult.timedOut).toBe(false);
        expect(exitResult).toMatchObject({ code: 0, signal: null });

        const releaseDeadline = Date.now() + 2_000;
        let portIsOpen = true;
        while (portIsOpen && Date.now() < releaseDeadline) {
          portIsOpen = await new Promise<boolean>((resolve) => {
            const socket = net.connect(fixturePort!, "127.0.0.1");
            socket.setTimeout(250);
            socket.once("connect", () => {
              socket.destroy();
              resolve(true);
            });
            socket.once("error", () => resolve(false));
            socket.once("timeout", () => {
              socket.destroy();
              resolve(false);
            });
          });
          if (portIsOpen) await delay(50);
        }
        expect(portIsOpen).toBe(false);
      } finally {
        if (supervisor.exitCode === null) supervisor.kill("SIGKILL");
        if (fixturePid) {
          try {
            process.kill(-fixturePid, "SIGKILL");
          } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
          }
        }
      }
    },
    15_000,
  );

  it("shows help with --help flag", async () => {
    const child = spawn(
      process.execPath,
      ["scripts/dev-with-automation.mjs", "--help"],
      {
        cwd: repoRoot,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });

    const [code] = await once(child, "exit");

    expect(code).toBe(0);
    expect(output).toContain("Agent Canvas + Automation Development Stack");
    expect(output).toContain("--port");
    expect(output).toContain("--automation-ref");
    expect(output).toContain("--automation-repo");
    expect(output).toContain("--static");
    expect(output).toContain("--dynamic");
    expect(output).toContain("--frontend-only");
    expect(output).toContain("--backend-only");
    expect(output).toContain("OH_AUTOMATION_GIT_REF");
    expect(output).toContain("OH_AGENT_SERVER_LOCAL_PATH");
    expect(output).toContain("OPENHANDS_AUTOMATION_API_KEY");
    expect(output).toContain("SECRETS:");
  });

  it("fails fast with a clear error when OH_AGENT_SERVER_LOCAL_PATH is invalid", async () => {
    // Arrange: an absolute but empty directory — `validateLocalAgentServerPath`
    // requires the four workspace subdirs (openhands-agent-server, openhands-sdk,
    // openhands-tools, openhands-workspace) and must reject this.
    const emptyDir = mkdtempSync(path.join(tmpdir(), "bad-sdk-"));

    // Stub a no-op `uvx` on PATH so `checkPrerequisites` passes even on CI
    // runners that don't have uv installed. The prerequisite check must
    // succeed so the LOCAL_PATH validation guard (the actual subject of this
    // test, which runs immediately after) is exercised.
    const isWindows = process.platform === "win32";
    const stubBinDir = mkdtempSync(path.join(tmpdir(), "stub-bin-"));
    if (isWindows) {
      writeFileSync(path.join(stubBinDir, "uvx.cmd"), "@exit /b 0\r\n");
    } else {
      writeFileSync(path.join(stubBinDir, "uvx"), "#!/bin/sh\nexit 0\n", {
        mode: 0o755,
      });
    }

    // Act: spawn `dev-with-automation.mjs` with that path set. Stubbed `uvx`
    // is prepended to PATH so the prerequisite check passes; the validation
    // guard must trip *after* those checks but *before* port allocation, so
    // we can assert on both the error message and the absence of side effects.
    const child = spawn(process.execPath, ["scripts/dev-with-automation.mjs"], {
      cwd: repoRoot,
      env: {
        PATH: `${stubBinDir}${path.delimiter}${process.env.PATH ?? ""}`,
        HOME: process.env.HOME ?? "",
        OH_AGENT_SERVER_LOCAL_PATH: emptyDir,
        // Windows needs these for `where.exe` to resolve the stub and npm
        ...(isWindows
          ? {
              PATHEXT: process.env.PATHEXT ?? ".CMD;.EXE;.BAT;.COM",
              SystemRoot: process.env.SystemRoot ?? "",
              USERPROFILE: process.env.USERPROFILE ?? "",
            }
          : {}),
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      output += chunk.toString();
    });

    const exitResult = await Promise.race([
      once(child, "exit").then(([code, signal]) => ({
        code,
        signal,
        timedOut: false,
      })),
      delay(10_000).then(() => ({ code: null, signal: null, timedOut: true })),
    ]);

    if (exitResult.timedOut) {
      child.kill("SIGKILL");
    }

    try {
      // Assert: process exits non-zero, surfaces the validator's error, and
      // never reaches `buildConfig` (no `[ports] Allocating ports...` log).
      expect(exitResult.timedOut).toBe(false);
      expect(exitResult.code).toBe(1);
      expect(output).toContain(
        "OH_AGENT_SERVER_LOCAL_PATH is missing expected workspace package",
      );
      expect(output).not.toContain("Allocating ports");
    } finally {
      rmSync(emptyDir, { recursive: true, force: true });
      rmSync(stubBinDir, { recursive: true, force: true });
    }
  });

  it("exits promptly when uvx is missing", async () => {
    const child = spawn(process.execPath, ["scripts/dev-with-automation.mjs"], {
      cwd: repoRoot,
      env: {
        PATH: "",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      output += chunk.toString();
    });

    const exitResult = await Promise.race([
      once(child, "exit").then(([code, signal]) => ({
        code,
        signal,
        timedOut: false,
      })),
      delay(4_000).then(() => ({ code: null, signal: null, timedOut: true })),
    ]);

    if (exitResult.timedOut) {
      child.kill("SIGKILL");
    }

    expect(exitResult.timedOut).toBe(false);
    expect(exitResult.code).toBe(1);
    expect(output).toContain("uvx");
  });
});
