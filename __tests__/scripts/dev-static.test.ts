import { spawn } from "node:child_process";
import { once } from "node:events";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";

import { buildAutomationBackendEnv } from "../../scripts/dev-static.mjs";

describe("dev-static", () => {
  it("uses the same session key for both agent-server and automation backend auth", () => {
    const env = buildAutomationBackendEnv(
      {
        agentServerPort: 18000,
        ingressPort: 8000,
        sessionApiKey: "shared-session-key",
        stateDir: "/tmp/agent-canvas-state",
      },
      {},
    );

    // Both backends receive the same key value
    expect(env).toMatchObject({
      AUTOMATION_AGENT_SERVER_URL: "http://localhost:18000",
      AUTOMATION_AGENT_SERVER_API_KEY: "shared-session-key",
      AUTOMATION_LOCAL_API_KEY: "shared-session-key",
      AUTOMATION_POSTHOG_API_KEY:
        "phc_kBtz5nKmxVRRQ7HtPwr2QX9eMC5j65zE86QKocVNwb4U",
      AUTOMATION_POSTHOG_HOST: "https://us.i.posthog.com",
    });
  });
});

describe("dev-static signal handler ownership", () => {
  // This suite imports dev-static.mjs for buildAutomationBackendEnv above, which
  // is precisely the case that must not take over the importer's signals: the
  // handlers would be wired to dev-static's own process registry, empty in a
  // vitest worker, and its shutdown() calls process.exit on a 3s timer.
  it("does not install signal handlers when the module is only imported", async () => {
    const repoRoot = path.resolve(
      path.dirname(fileURLToPath(import.meta.url)),
      "../..",
    );
    const moduleUrl = pathToFileURL(
      path.join(repoRoot, "scripts", "dev-static.mjs"),
    ).href;

    const child = spawn(
      process.execPath,
      [
        "--input-type=module",
        "--eval",
        [
          'import { writeSync } from "node:fs";',
          'const signals = ["SIGINT", "SIGTERM", "SIGHUP"];',
          "const counts = () =>",
          "  Object.fromEntries(signals.map((s) => [s, process.listenerCount(s)]));",
          "const before = counts();",
          `await import(${JSON.stringify(moduleUrl)});`,
          'writeSync(1, "COUNTS " + JSON.stringify({ before, after: counts() }) + "\\n");',
        ].join("\n"),
      ],
      { cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"] },
    );

    let output = "";
    child.stdout.on("data", (chunk: Buffer) => (output += chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => (output += chunk.toString()));
    // "close" rather than "exit": exit can fire before the stdio pipes are
    // drained, and this test parses the child's stdout for its result.
    await once(child, "close");

    const match = output.match(/COUNTS (\{.*\})/);
    expect(match, output).not.toBeNull();
    expect(JSON.parse(match![1])).toEqual({
      before: { SIGINT: 0, SIGTERM: 0, SIGHUP: 0 },
      after: { SIGINT: 0, SIGTERM: 0, SIGHUP: 0 },
    });
  });
});
