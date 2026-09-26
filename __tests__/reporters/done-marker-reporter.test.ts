import { existsSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TestCase, TestResult } from "@playwright/test/reporter";

import DoneMarkerReporter from "#/../tests/e2e/mock-llm/reporters/done-marker-reporter";

// Mirrors the reporter's marker directory (the reporter derives it from cwd).
const markerDir = join(process.cwd(), ".mock-llm-markers");
const testsDonePath = join(markerDir, ".tests-done");
const resultsPath = join(markerDir, ".results.json");

function makeTestCase(title: string, retries = 0): TestCase {
  return {
    title,
    titlePath: () => ["suite", title],
    retries,
    results: [],
  } as unknown as TestCase;
}

function makeResult(status: TestResult["status"], retries = 0): TestResult {
  return {
    status,
    duration: 100,
    errors: [],
    retry: retries,
  } as unknown as TestResult;
}

describe("DoneMarkerReporter", () => {
  let reporter: DoneMarkerReporter;

  beforeEach(() => {
    vi.resetModules();
    reporter = new DoneMarkerReporter();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    for (const p of [resultsPath, testsDonePath]) {
      if (existsSync(p)) {
        rmSync(p);
      }
    }
  });

  it("counts a flaky test once (final attempt only), not once per retry", () => {
    const suite = { allTests: () => [makeTestCase("flaky")] };
    reporter.onBegin(undefined, suite as never);

    // First attempt fails (flaky), second attempt passes.
    const test = makeTestCase("flaky");
    const failResult = makeResult("failed", 0);
    const passResult = makeResult("passed", 1);
    test.results = [failResult, passResult];

    reporter.onTestEnd(test, failResult);
    // An earlier attempt must not be counted or flushed any markers yet.
    expect(existsSync(testsDonePath)).toBe(false);

    reporter.onTestEnd(test, passResult);
    // Only the final attempt completes the suite — exactly once.
    expect(readFileSync(testsDonePath, "utf8")).toBe("passed");
    expect(JSON.parse(readFileSync(resultsPath, "utf8"))).toMatchObject({
      completed: 1,
      total: 1,
    });
  });
});
