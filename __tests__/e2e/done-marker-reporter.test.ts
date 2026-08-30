// @vitest-environment node
/**
 * Unit tests for DoneMarkerReporter (tests/e2e/mock-llm/reporters/).
 *
 * Regression coverage for https://github.com/OpenHands/OpenHands/issues/16977:
 * the reporter counted test ATTEMPTS against the number of test CASES, so a
 * single flaky retry could end the mock-LLM Docker E2E run early (marker
 * written before queued tests ran) and report `failed` for a run Playwright
 * considers green (allPassed cleared by the first failed attempt and never
 * reset when the retry passed).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  FullResult,
  TestCase,
  TestResult,
} from "@playwright/test/reporter";

// The reporter imports `mkdirSync`/`writeFileSync` from node:fs and writes
// markers under process.cwd(). Mock fs so tests never touch the real repo.
vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return {
    ...actual,
    mkdirSync: vi.fn(),
    writeFileSync: vi.fn(),
  };
});

// Import AFTER vi.mock so the reporter gets the mocked fs bindings.
import { writeFileSync } from "node:fs";
import DoneMarkerReporter from "../../tests/e2e/mock-llm/reporters/done-marker-reporter";

type MockWriteFileSync = ReturnType<typeof vi.fn>;

interface FakeAttempt {
  status: string; // "passed" | "failed" | "timedOut" | "skipped"
  retry: number;
  duration: number;
  errors: string[];
}

/** Mirrors Playwright's TestCase.outcome() over the attempts so far. */
function outcomeOf(statuses: string[]): string {
  const firstNonSkipped = statuses.find((s) => s !== "skipped");
  if (!firstNonSkipped) return "skipped";
  if (firstNonSkipped === "passed") return "expected";
  if (statuses.some((s) => s === "passed")) return "flaky";
  return "unexpected";
}

/** Fake TestCase whose results/outcome grow as attempts are pushed. */
function makeTest(id: string, retries: number) {
  const statuses: string[] = [];
  return {
    id,
    retries,
    get results() {
      return statuses.map((s) => ({ status: s, expectedStatus: "passed" }));
    },
    outcome: () => outcomeOf(statuses),
    titlePath: () => ["mock-llm", id],
    pushAttempt(status: string) {
      statuses.push(status);
    },
  };
}

function makeResult(attempt: FakeAttempt): TestResult {
  return {
    retry: attempt.retry,
    status: attempt.status,
    duration: attempt.duration,
    errors: attempt.errors.map((message) => ({ message })),
  } as unknown as TestResult;
}

function writtenContent(write: MockWriteFileSync, name: string): string {
  const call = write.mock.calls
    .filter((c) => String(c[0]).endsWith(name))
    .at(-1); // the reporter flushes .results.json after every test — use the latest
  expect(call, `expected a write to ${name}`).toBeTruthy();
  return String(call![1]);
}

function writeCount(write: MockWriteFileSync, name: string): number {
  return write.mock.calls.filter((c) => String(c[0]).endsWith(name)).length;
}

/** Parse the marker dir's .results.json written by the reporter. */
function lastResultsJson(write: MockWriteFileSync): {
  status: string;
  completed: number;
  total: number;
  tests: { title: string; status: string }[];
} {
  return JSON.parse(writtenContent(write, ".results.json"));
}

describe("DoneMarkerReporter", () => {
  let write: MockWriteFileSync;

  beforeEach(() => {
    write = writeFileSync as unknown as MockWriteFileSync;
    write.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("counts each test case once and marks the run passed when all pass", () => {
    const reporter = new DoneMarkerReporter();
    const a = makeTest("a", 0);
    const b = makeTest("b", 0);
    reporter.onBegin(undefined, {
      allTests: () => [a, b],
    } as unknown as Parameters<typeof reporter.onBegin>[1]);

    a.pushAttempt("passed");
    reporter.onTestEnd(
      a as unknown as TestCase,
      makeResult({ status: "passed", retry: 0, duration: 10, errors: [] }),
    );
    b.pushAttempt("passed");
    reporter.onTestEnd(
      b as unknown as TestCase,
      makeResult({ status: "passed", retry: 0, duration: 20, errors: [] }),
    );

    const results = lastResultsJson(write);
    expect(results.completed).toBe(2);
    expect(results.total).toBe(2);
    expect(results.status).toBe("passed");
    expect(results.tests).toHaveLength(2);
    expect(writtenContent(write, ".tests-done")).toBe("passed");
    expect(writeCount(write, ".all-passed")).toBe(1);
  });

  it("does not write the completion marker while later cases are still queued (flaky mid-suite)", () => {
    const reporter = new DoneMarkerReporter();
    const a = makeTest("a", 0);
    const flaky = makeTest("flaky", 1);
    const z = makeTest("z", 0);
    reporter.onBegin(undefined, {
      allTests: () => [a, flaky, z],
    } as unknown as Parameters<typeof reporter.onBegin>[1]);

    // Case a passes.
    a.pushAttempt("passed");
    reporter.onTestEnd(
      a as unknown as TestCase,
      makeResult({ status: "passed", retry: 0, duration: 10, errors: [] }),
    );

    // Flaky case: first attempt fails and is retried. The reporter must NOT
    // count it yet, and must NOT write .tests-done (case z is still queued).
    flaky.pushAttempt("failed");
    reporter.onTestEnd(
      flaky as unknown as TestCase,
      makeResult({
        status: "failed",
        retry: 0,
        duration: 50,
        errors: ["boom"],
      }),
    );
    expect(writeCount(write, ".tests-done")).toBe(0);
    expect(writeCount(write, ".all-passed")).toBe(0);

    // Retry passes — still one more queued case, so no completion marker.
    flaky.pushAttempt("passed");
    reporter.onTestEnd(
      flaky as unknown as TestCase,
      makeResult({ status: "passed", retry: 1, duration: 30, errors: [] }),
    );
    expect(writeCount(write, ".tests-done")).toBe(0);
    expect(lastResultsJson(write).completed).toBe(2); // a + flaky, not attempts

    // Last case finishes — only now are the completion markers written.
    z.pushAttempt("passed");
    reporter.onTestEnd(
      z as unknown as TestCase,
      makeResult({ status: "passed", retry: 0, duration: 5, errors: [] }),
    );

    const results = lastResultsJson(write);
    expect(results.completed).toBe(3);
    expect(results.total).toBe(3);
    expect(results.status).toBe("passed");
    // One entry per case — the flaky case appears once, as its passing retry.
    expect(results.tests.map((t) => t.status)).toEqual([
      "passed",
      "passed",
      "passed",
    ]);
    expect(writtenContent(write, ".tests-done")).toBe("passed");
    expect(writeCount(write, ".all-passed")).toBe(1);
  });

  it("treats a flaky test that passes on retry as green when it is the last case", () => {
    const reporter = new DoneMarkerReporter();
    const a = makeTest("a", 0);
    const lastFlaky = makeTest("last-flaky", 1);
    reporter.onBegin(undefined, {
      allTests: () => [a, lastFlaky],
    } as unknown as Parameters<typeof reporter.onBegin>[1]);

    a.pushAttempt("passed");
    reporter.onTestEnd(
      a as unknown as TestCase,
      makeResult({ status: "passed", retry: 0, duration: 10, errors: [] }),
    );

    lastFlaky.pushAttempt("failed");
    reporter.onTestEnd(
      lastFlaky as unknown as TestCase,
      makeResult({
        status: "failed",
        retry: 0,
        duration: 50,
        errors: ["boom"],
      }),
    );
    // Even though completed (1) >= total (2) is false, the failed attempt must
    // not be counted; and no marker may appear for a mid-retry state.
    expect(writeCount(write, ".tests-done")).toBe(0);

    lastFlaky.pushAttempt("passed");
    reporter.onTestEnd(
      lastFlaky as unknown as TestCase,
      makeResult({ status: "passed", retry: 1, duration: 30, errors: [] }),
    );

    const results = lastResultsJson(write);
    expect(results.completed).toBe(2);
    expect(results.total).toBe(2);
    expect(results.tests.map((t) => t.status)).toEqual(["passed", "passed"]);
    expect(writtenContent(write, ".tests-done")).toBe("passed");
    expect(writeCount(write, ".all-passed")).toBe(1);
  });

  it("writes failed and no .all-passed when a test fails every attempt", () => {
    const reporter = new DoneMarkerReporter();
    const bad = makeTest("bad", 1);
    reporter.onBegin(undefined, {
      allTests: () => [bad],
    } as unknown as Parameters<typeof reporter.onBegin>[1]);

    bad.pushAttempt("failed");
    reporter.onTestEnd(
      bad as unknown as TestCase,
      makeResult({ status: "failed", retry: 0, duration: 40, errors: ["x"] }),
    );
    expect(writeCount(write, ".tests-done")).toBe(0);

    bad.pushAttempt("failed");
    reporter.onTestEnd(
      bad as unknown as TestCase,
      makeResult({ status: "failed", retry: 1, duration: 45, errors: ["x"] }),
    );

    const results = lastResultsJson(write);
    expect(results.completed).toBe(1);
    expect(results.status).toBe("failed");
    expect(writtenContent(write, ".tests-done")).toBe("failed");
    expect(writeCount(write, ".all-passed")).toBe(0);
  });

  it("never reports completed greater than total across retries", () => {
    const reporter = new DoneMarkerReporter();
    const flaky = makeTest("flaky", 2);
    reporter.onBegin(undefined, {
      allTests: () => [flaky],
    } as unknown as Parameters<typeof reporter.onBegin>[1]);

    flaky.pushAttempt("failed");
    reporter.onTestEnd(
      flaky as unknown as TestCase,
      makeResult({ status: "failed", retry: 0, duration: 40, errors: ["x"] }),
    );
    flaky.pushAttempt("failed");
    reporter.onTestEnd(
      flaky as unknown as TestCase,
      makeResult({ status: "failed", retry: 1, duration: 40, errors: ["x"] }),
    );
    flaky.pushAttempt("passed");
    reporter.onTestEnd(
      flaky as unknown as TestCase,
      makeResult({ status: "passed", retry: 2, duration: 20, errors: [] }),
    );

    const results = lastResultsJson(write);
    expect(results.completed).toBe(1);
    expect(results.completed).toBeLessThanOrEqual(results.total);
    expect(writtenContent(write, ".tests-done")).toBe("passed");
  });

  it("keeps the onEnd fallback: no test ran means failed with whatever exists", () => {
    const reporter = new DoneMarkerReporter();
    reporter.onEnd({ status: "failed" } as FullResult);

    const results = lastResultsJson(write);
    expect(results.status).toBe("failed");
    expect(writtenContent(write, ".tests-done")).toBe("failed");
    expect(writeCount(write, ".all-passed")).toBe(0);
  });
});
