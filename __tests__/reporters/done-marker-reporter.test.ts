import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TestCase, TestResult } from "@playwright/test/reporter";

import DoneMarkerReporter from "#/../tests/e2e/mock-llm/reporters/done-marker-reporter";

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
  const markerDir = "/tmp/openhands-marker-test";
  let reporter: DoneMarkerReporter;

  beforeEach(() => {
    vi.resetModules();
    process.env.MARKER_DIR = markerDir;
    reporter = new DoneMarkerReporter();
  });

  afterEach(() => {
    vi.restoreAllMocks();
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
    reporter.onTestEnd(test, passResult);

    // After the final attempt the suite is done exactly once.
    expect(reporter.completedTests).toBe(1);
  });
});
