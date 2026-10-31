// @vitest-environment node
import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const playgrounds: string[] = [];
const reporter = resolve(
  "tests/e2e/mock-llm/reporters/done-marker-reporter.ts",
);
const teardown = resolve(
  "tests/e2e/mock-llm/reporters/done-marker-teardown.ts",
);
const playwright = resolve("node_modules/playwright/cli.js");
const testModule = resolve("node_modules/@playwright/test/index.mjs");

function runSuite(
  source: string,
  options: Record<string, unknown> = {},
  files: Record<string, string> = {},
) {
  const cwd = mkdtempSync(join(tmpdir(), "mock-llm-marker-"));
  playgrounds.push(cwd);
  writeFileSync(
    join(cwd, "probe.spec.mjs"),
    `import { test, expect } from ${JSON.stringify(pathToFileURL(testModule).href)};
import { existsSync, appendFileSync } from 'node:fs';
${source}`,
  );
  for (const [name, content] of Object.entries(files))
    writeFileSync(join(cwd, name), content);
  writeFileSync(
    join(cwd, "playwright.config.mjs"),
    `const config = ${JSON.stringify({
      testDir: cwd,
      testMatch: "probe.spec.mjs",
      workers: 1,
      retries: 0,
      timeout: 10_000,
      reporter: [[reporter]],
      globalTeardown: teardown,
      ...options,
    })};
if (config.webServer?.wait) config.webServer.wait.stdout = /ready/;
export default config;`,
  );
  const run = spawnSync(
    process.execPath,
    [playwright, "test", "--config", join(cwd, "playwright.config.mjs")],
    { cwd, encoding: "utf8", timeout: 25_000 },
  );
  expect(run.error, run.stdout + run.stderr).toBeUndefined();
  const marker = (name: string) => join(cwd, ".mock-llm-markers", name);
  return {
    exitCode: run.status,
    done: readFileSync(marker(".tests-done"), "utf8"),
    passed: existsSync(marker(".all-passed")),
    results: JSON.parse(readFileSync(marker(".results.json"), "utf8")),
    read: (name: string) => readFileSync(join(cwd, name), "utf8"),
  };
}

afterEach(() => {
  for (const cwd of playgrounds.splice(0))
    rmSync(cwd, { recursive: true, force: true });
});

describe("mock-LLM completion markers with real Playwright scheduling", () => {
  it.each(["default", "serial"])(
    "waits for the final retry and reports one result per case in %s mode",
    (mode) => {
      const run = runSuite(
        `test.describe.configure({ mode: '${mode}' });
test.beforeEach(() => appendFileSync('markers.jsonl', JSON.stringify(existsSync('.mock-llm-markers/.tests-done')) + '\\n'));
test('first', () => {});
${mode === "serial" ? "test('flaky', ({}, info) => expect(info.retry).toBe(1)); test('last', () => {});" : "test('middle', () => {}); test('last flaky', ({}, info) => expect(info.retry).toBe(1));"}`,
        { retries: 1 },
      );

      expect(run.exitCode).toBe(0);
      expect(
        run
          .read("markers.jsonl")
          .trim()
          .split("\n")
          .map((line) => JSON.parse(line)),
      ).not.toContain(true);
      expect(run.done).toBe("passed");
      expect(run.passed).toBe(true);
      expect(run.results).toMatchObject({
        status: "passed",
        completed: 3,
        total: 3,
      });
      expect(run.results.tests).toHaveLength(3);
      expect(run.results.tests[mode === "serial" ? 1 : 2]).toMatchObject({
        status: "passed",
        error: "",
      });
    },
    30_000,
  );

  it("keeps a permanent failure red after its retry", () => {
    const run = runSuite(
      "test('fails twice', () => expect(false).toBe(true));",
      { retries: 1 },
    );
    expect(run.exitCode).toBe(1);
    expect(run.done).toBe("failed");
    expect(run.passed).toBe(false);
    expect(run.results.tests).toHaveLength(1);
    expect(run.results.tests[0].status).toBe("failed");
  }, 30_000);

  it("does not rescue a run cut short by globalTimeout", () => {
    const run = runSuite(
      `test('passes', () => {});
test('interrupted', async () => { await new Promise(r => setTimeout(r, 20000)); });
test('never starts', () => {});`,
      { globalTimeout: 3000 },
    );

    expect(run.exitCode).not.toBe(0);
    expect(run.done).toBe("failed");
    expect(run.passed).toBe(false);
    expect(run.results.completed).toBeLessThan(run.results.total);
  }, 30_000);

  it.each([false, true])(
    "honors expected failures (unexpected pass=%s)",
    (unexpectedPass) => {
      const run = runSuite(
        `test('expected failure', () => { test.fail(); expect(${unexpectedPass}).toBe(true); });
test.skip('deliberately skipped', () => {});`,
      );

      expect(run.exitCode).toBe(unexpectedPass ? 1 : 0);
      expect(run.done).toBe(unexpectedPass ? "failed" : "passed");
      expect(run.passed).toBe(!unexpectedPass);
      expect(run.results.tests[0].status).toBe(
        unexpectedPass ? "failed" : "passed",
      );
    },
    30_000,
  );

  it.each(["before", "after"])(
    "fails global errors %s marker finalization",
    (timing) => {
      const failingHook =
        "export default () => { throw new Error('global cleanup failed'); };";
      const finishHooks = [teardown];
      const run = runSuite(
        "test('passes', () => {});",
        {
          globalTeardown:
            timing === "before"
              ? ["./failure.mjs", ...finishHooks]
              : [...finishHooks, "./failure.mjs"],
        },
        { "failure.mjs": failingHook },
      );

      expect(run.exitCode).toBe(1);
      expect(run.done).toBe("failed");
      expect(run.passed).toBe(false);
    },
    30_000,
  );

  it("fails startup errors even when no test ever reports", () => {
    const run = runSuite(
      "test('never starts', () => {});",
      { globalSetup: "./failure.mjs" },
      {
        "failure.mjs":
          "export default () => { throw new Error('setup failed'); };",
      },
    );

    expect(run.exitCode).toBe(1);
    expect(run.done).toBe("failed");
    expect(run.passed).toBe(false);
  }, 30_000);

  it.skipIf(process.platform === "win32")(
    "preserves passing markers through a webServer teardown timeout",
    () => {
      const run = runSuite(
        "test('passes', () => {}); test('also passes', () => {});",
        {
          globalTimeout: 4000,
          webServer: {
            command: "node server.cjs",
            // Readiness via stdout exercises the real webServer plugin without
            // binding a socket or starting a browser/backend.
            wait: { stdout: "ready" },
            gracefulShutdown: { signal: "SIGTERM", timeout: 5000 },
          },
        },
        {
          "server.cjs": `const fs = require('node:fs');
setInterval(() => {}, 1000);
console.log('ready');
process.on('SIGTERM', () => {
  fs.writeFileSync('shutdown.json', JSON.stringify({
    done: fs.existsSync('.mock-llm-markers/.tests-done') ? fs.readFileSync('.mock-llm-markers/.tests-done', 'utf8') : null,
    passed: fs.existsSync('.mock-llm-markers/.all-passed')
  }));
});`,
        },
      );

      expect(run.exitCode).toBe(1);
      expect(JSON.parse(run.read("shutdown.json"))).toEqual({
        done: "passed",
        passed: true,
      });
      expect(run.done).toBe("passed");
      expect(run.passed).toBe(true);
    },
    30_000,
  );
});
