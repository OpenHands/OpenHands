// @vitest-environment node
import { spawnSync } from "node:child_process";
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
  existsSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

const playgrounds: string[] = [];
const playwright = resolve("node_modules/@playwright/test/cli.js");
const reporter = resolve(
  "tests/e2e/mock-llm/reporters/done-marker-reporter.ts",
);
const testModule = resolve("node_modules/@playwright/test/index.mjs");

function runSuite(source: string, options = {}) {
  const cwd = mkdtempSync(join(tmpdir(), "done-marker-"));
  playgrounds.push(cwd);
  writeFileSync(
    join(cwd, "suite.spec.mjs"),
    `import { test, expect } from ${JSON.stringify(testModule)};\n${source}`,
  );
  writeFileSync(
    join(cwd, "teardown.mjs"),
    `import { existsSync, readFileSync, writeFileSync } from 'node:fs';
export default () => { writeFileSync('before-onEnd.json', JSON.stringify({
  done: existsSync('.mock-llm-markers/.tests-done') ? readFileSync('.mock-llm-markers/.tests-done', 'utf8') : null,
  passed: existsSync('.mock-llm-markers/.all-passed')
})); };`,
  );
  writeFileSync(
    join(cwd, "playwright.config.mjs"),
    `export default ${JSON.stringify({
      testDir: cwd,
      workers: 1,
      retries: 0,
      timeout: 5000,
      reporter: [[reporter]],
      globalTeardown: join(cwd, "teardown.mjs"),
      ...options,
    })};`,
  );
  const result = spawnSync(
    process.execPath,
    [playwright, "test", "--config", join(cwd, "playwright.config.mjs")],
    {
      cwd,
      encoding: "utf8",
      timeout: 20000,
    },
  );
  expect(result.error, result.stderr).toBeUndefined();
  const marker = (name: string) => join(cwd, ".mock-llm-markers", name);
  return {
    exitCode: result.status,
    done: readFileSync(marker(".tests-done"), "utf8"),
    passed: existsSync(marker(".all-passed")),
    results: JSON.parse(readFileSync(marker(".results.json"), "utf8")),
    beforeEnd: existsSync(join(cwd, "before-onEnd.json"))
      ? JSON.parse(readFileSync(join(cwd, "before-onEnd.json"), "utf8"))
      : null,
  };
}

afterEach(() => {
  for (const cwd of playgrounds.splice(0))
    rmSync(cwd, { recursive: true, force: true });
});

describe("DoneMarkerReporter with real Playwright runs", () => {
  it("writes passing markers before teardown and onEnd", () => {
    const run = runSuite(
      `test('passes', () => {}); test('also passes', () => {});`,
    );
    expect(run.exitCode).toBe(0);
    expect(run.beforeEnd).toEqual({ done: "passed", passed: true });
    expect(run.done).toBe("passed");
    expect(run.passed).toBe(true);
  }, 30000);

  it("does not count retries as completed tests when globalTimeout truncates the suite", () => {
    const run = runSuite(
      `
      test('retries', ({}, info) => { expect(info.retry).toBe(1); });
      test('interrupted', async () => { await new Promise(r => setTimeout(r, 20000)); });
      test('never started', () => {});
    `,
      { retries: 1, globalTimeout: 8000, timeout: 15000 },
    );
    expect(run.exitCode).not.toBe(0);
    expect(run.beforeEnd).toBeNull();
    expect(run.done).toBe("failed");
    expect(run.passed).toBe(false);
    expect(run.results.completed).toBeLessThan(run.results.total);
    expect(run.results.status).toBe("in_progress");
  }, 30000);

  it("writes failing markers before teardown for a completed failing suite", () => {
    const run = runSuite(`test('fails', () => { expect(true).toBe(false); });`);
    expect(run.exitCode).toBe(1);
    expect(run.beforeEnd).toEqual({ done: "failed", passed: false });
    expect(run.done).toBe("failed");
    expect(run.passed).toBe(false);
  }, 30000);
});
