// @vitest-environment node
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { TestCase, TestResult } from "@playwright/test/reporter";
import {
  beforeAll,
  afterAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

/**
 * `DoneMarkerReporter` writes the markers the mock-LLM workflows poll on, so
 * these tests drive it through real attempt sequences and read the files back.
 *
 * The sequences are not invented. The serial-mode one below is transcribed
 * from a probe of the Playwright version this repo pins, because the
 * interesting behaviour is Playwright's scheduling rather than anything the
 * reporter can be reasoned about in isolation.
 *
 * The marker directory is resolved from `process.cwd()` when the module loads,
 * so the reporter is imported once with `cwd` pointed at a temp directory —
 * real writes, nothing touched in the checkout.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
let DoneMarkerReporter: any;
let root: string;
let markerDir: string;

beforeAll(async () => {
  root = mkdtempSync(join(tmpdir(), "done-marker-reporter-"));
  markerDir = join(root, ".mock-llm-markers");

  const cwd = vi.spyOn(process, "cwd").mockReturnValue(root);
  ({ default: DoneMarkerReporter } =
    await import("../../tests/e2e/mock-llm/reporters/done-marker-reporter"));
  cwd.mockRestore();
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

beforeEach(() => {
  rmSync(markerDir, { recursive: true, force: true });
});

// ── Fixtures ───────────────────────────────────────────────────────────

type Status = "passed" | "failed" | "timedOut" | "skipped" | "interrupted";

/** A test case as the reporter reads it: an id, a title path, a retry budget. */
function testCase(id: string, title: string, retries = 1): TestCase {
  return {
    id,
    retries,
    titlePath: () => ["", "chromium", "flaky.spec.ts", title],
  } as unknown as TestCase;
}

/** One attempt at a case. `retry` is 0 for the first attempt, 1 for the first retry. */
function attempt(status: Status, retry: number, error?: string): TestResult {
  return {
    status,
    retry,
    duration: 120,
    errors: error ? [{ message: error }] : [],
  } as unknown as TestResult;
}

// ── Marker readers ─────────────────────────────────────────────────────

const marker = (name: string): string | null => {
  const path = join(markerDir, name);
  return existsSync(path) ? readFileSync(path, "utf8") : null;
};

const results = () =>
  JSON.parse(readFileSync(join(markerDir, ".results.json"), "utf8"));

/** Start a reporter over `cases`, mimicking Playwright's `onBegin` contract. */
function begin(cases: TestCase[]) {
  const reporter = new DoneMarkerReporter();
  reporter.onBegin({}, { allTests: () => cases });
  return reporter;
}

describe("DoneMarkerReporter", () => {
  describe("serial mode — the shape every mock-LLM spec runs in", () => {
    /**
     * Probed from the real runner with
     * `test.describe.configure({ mode: "serial" })` and `retries: 1`, one
     * flaky case in the middle:
     *
     *   [1] t1 passed  retry=0     [4] t1 passed retry=1
     *   [2] t2 failed  retry=0     [5] t2 passed retry=1
     *   [3] t3 skipped retry=0     [6] t3 passed retry=1
     *
     * A serial-group failure re-runs the whole group, so an attempt reporting
     * `passed` or `skipped` is not evidence that its case is finished.
     */
    const cases = () => [
      testCase("t1", "t1 - passes"),
      testCase("t2", "t2 - flaky"),
      testCase("t3", "t3 - last"),
    ];

    const firstPass = (reporter: any, [t1, t2, t3]: TestCase[]) => {
      reporter.onTestEnd(t1, attempt("passed", 0));
      reporter.onTestEnd(t2, attempt("failed", 0, "simulated flake"));
      reporter.onTestEnd(t3, attempt("skipped", 0));
    };

    it("writes no completion marker while the group is being re-run", () => {
      const group = cases();
      const reporter = begin(group);
      firstPass(reporter, group);

      // Every case has reported once, but the group retry has not started.
      expect(marker(".tests-done")).toBeNull();
      expect(marker(".all-passed")).toBeNull();

      reporter.onTestEnd(group[0], attempt("passed", 1));
      reporter.onTestEnd(group[1], attempt("passed", 1));

      // t3's retry is still queued. This is the point where counting attempts,
      // or classifying `passed`/`skipped` as terminal, wrote the marker early
      // and let the workflow kill a run that had work left.
      expect(marker(".tests-done")).toBeNull();
      expect(marker(".all-passed")).toBeNull();
      expect(results().status).toBe("in_progress");
    });

    it("reports the flaky group as passed once the run ends", () => {
      const group = cases();
      const reporter = begin(group);
      firstPass(reporter, group);
      reporter.onTestEnd(group[0], attempt("passed", 1));
      reporter.onTestEnd(group[1], attempt("passed", 1));
      reporter.onTestEnd(group[2], attempt("passed", 1));
      reporter.onEnd({ status: "passed" });

      expect(marker(".tests-done")).toBe("passed");
      expect(marker(".all-passed")).toBe("1");
      expect(results()).toMatchObject({
        status: "passed",
        completed: 3,
        total: 3,
      });
    });

    it("keeps the retry's verdict, not the failed first attempt's", () => {
      const group = cases();
      const reporter = begin(group);
      firstPass(reporter, group);
      reporter.onTestEnd(group[0], attempt("passed", 1));
      reporter.onTestEnd(group[1], attempt("passed", 1));
      reporter.onTestEnd(group[2], attempt("passed", 1));
      reporter.onEnd({ status: "passed" });

      const { tests } = results();
      expect(tests.map((t: { status: string }) => t.status)).toEqual([
        "passed",
        "passed",
        "passed",
      ]);
      expect(tests[1].error).toBe("");
    });
  });

  describe("a retried case is one case", () => {
    it("never reports more completed than total", () => {
      const [a, b] = [testCase("a", "a"), testCase("b", "b - flaky")];
      const reporter = begin([a, b]);

      const seen: { completed: number; total: number }[] = [];
      const record = () => {
        const { completed, total } = results();
        seen.push({ completed, total });
      };

      reporter.onTestEnd(a, attempt("passed", 0));
      record();
      reporter.onTestEnd(b, attempt("failed", 0, "flake"));
      record();
      reporter.onTestEnd(b, attempt("passed", 1));
      record();
      reporter.onEnd({ status: "passed" });
      record();

      expect(seen).toEqual([
        { completed: 1, total: 2 },
        { completed: 2, total: 2 },
        { completed: 2, total: 2 },
        { completed: 2, total: 2 },
      ]);
    });

    it("keeps completed and tests.length in step, so 'N not run' is honest", () => {
      // render-mock-llm-report.mjs derives "N not run" from total - completed
      // while rendering `tests`; the two must describe the same set.
      const [a, b, c] = [
        testCase("a", "a"),
        testCase("b", "b"),
        testCase("c", "c"),
      ];
      const reporter = begin([a, b, c]);

      reporter.onTestEnd(a, attempt("passed", 0));
      reporter.onTestEnd(b, attempt("failed", 0, "boom"));
      reporter.onTestEnd(b, attempt("failed", 1, "boom"));
      reporter.onEnd({ status: "failed" });

      const { completed, total, tests } = results();
      expect(completed).toBe(tests.length);
      expect(total - completed).toBe(1);
    });

    it("records one entry per case, carrying the final attempt's verdict", () => {
      const [a, b] = [testCase("a", "a"), testCase("b", "b - flaky")];
      const reporter = begin([a, b]);

      reporter.onTestEnd(a, attempt("passed", 0));
      reporter.onTestEnd(b, attempt("failed", 0, "simulated flake"));
      reporter.onTestEnd(b, attempt("passed", 1));
      reporter.onEnd({ status: "passed" });

      const { tests } = results();
      expect(tests).toHaveLength(2);
      // The failed attempt's error must not survive onto a case that passed —
      // render-mock-llm-report.mjs prints whatever is recorded here.
      expect(tests[1].error).toBe("");
      expect(tests[1].title).toBe("chromium › flaky.spec.ts › b - flaky");
    });
  });

  describe("genuine failures still fail", () => {
    it("writes failed and no .all-passed when a case fails every attempt", () => {
      const [a, b] = [testCase("a", "a"), testCase("b", "b - broken")];
      const reporter = begin([a, b]);

      reporter.onTestEnd(a, attempt("passed", 0));
      reporter.onTestEnd(b, attempt("failed", 0, "boom"));
      reporter.onTestEnd(b, attempt("failed", 1, "boom"));
      reporter.onEnd({ status: "failed" });

      expect(marker(".tests-done")).toBe("failed");
      expect(marker(".all-passed")).toBeNull();
      expect(results()).toMatchObject({ status: "failed", completed: 2 });
      expect(results().tests[1].error).toContain("boom");
    });

    it("treats a timed-out case as failed", () => {
      const a = testCase("a", "a - hangs");
      const reporter = begin([a]);

      reporter.onTestEnd(a, attempt("timedOut", 0));
      reporter.onTestEnd(a, attempt("timedOut", 1));
      reporter.onEnd({ status: "failed" });

      expect(marker(".tests-done")).toBe("failed");
      expect(marker(".all-passed")).toBeNull();
    });

    it("treats an interrupted case as failed", () => {
      // `interrupted` is in TestResult's status union and reaches the reporter
      // on SIGINT or maxFailures; it must not read as a pass.
      const [a, b] = [testCase("a", "a"), testCase("b", "b")];
      const reporter = begin([a, b]);

      reporter.onTestEnd(a, attempt("passed", 0));
      reporter.onTestEnd(b, attempt("interrupted", 0));
      reporter.onEnd({ status: "interrupted" });

      expect(marker(".tests-done")).toBe("failed");
      expect(marker(".all-passed")).toBeNull();
    });

    it("counts a deliberately skipped case as passing", () => {
      // `test.skip(condition, ...)` is used in the real specs, so a run whose
      // only non-passing case was skipped is green.
      const [a, b] = [testCase("a", "a"), testCase("b", "b - skipped")];
      const reporter = begin([a, b]);

      reporter.onTestEnd(a, attempt("passed", 0));
      reporter.onTestEnd(b, attempt("skipped", 0));
      reporter.onEnd({ status: "passed" });

      expect(marker(".tests-done")).toBe("passed");
      expect(marker(".all-passed")).toBe("1");
    });
  });

  describe("a truncated run is never green", () => {
    it("fails a run that ended with cases unreported", () => {
      // The global timeout or a kill lands here: three cases declared, two ran,
      // both passed. Reporting that as passed would hide the missing case.
      const [a, b, c] = [
        testCase("a", "a"),
        testCase("b", "b"),
        testCase("c", "c"),
      ];
      const reporter = begin([a, b, c]);

      reporter.onTestEnd(a, attempt("passed", 0));
      reporter.onTestEnd(b, attempt("passed", 0));
      reporter.onEnd({ status: "timedout" });

      expect(marker(".tests-done")).toBe("failed");
      expect(marker(".all-passed")).toBeNull();
      expect(results()).toMatchObject({ completed: 2, total: 3 });
    });

    it("fails a run where no test ever reported", () => {
      const reporter = begin([testCase("a", "a"), testCase("b", "b")]);

      reporter.onEnd({ status: "timedout" });

      expect(marker(".tests-done")).toBe("failed");
      expect(marker(".all-passed")).toBeNull();
      expect(results()).toMatchObject({ completed: 0, total: 2 });
    });

    it("fails a suite that declared no tests at all", () => {
      const reporter = begin([]);

      reporter.onEnd({ status: "failed" });

      expect(marker(".tests-done")).toBe("failed");
      expect(marker(".all-passed")).toBeNull();
      expect(results()).toMatchObject({ status: "failed", total: 0 });
    });

    it("writes no completion marker while tests are still running", () => {
      const [a, b] = [testCase("a", "a"), testCase("b", "b")];
      const reporter = begin([a, b]);

      reporter.onTestEnd(a, attempt("passed", 0));
      reporter.onTestEnd(b, attempt("passed", 0));

      // Every case has reported, but onEnd has not fired: the run is not over.
      expect(marker(".tests-done")).toBeNull();
      expect(marker(".all-passed")).toBeNull();
      expect(results().status).toBe("in_progress");
    });
  });

  describe("without retries, the npm lane behaves as before", () => {
    it("reports a first-attempt failure as failed", () => {
      const [a, b] = [testCase("a", "a", 0), testCase("b", "b", 0)];
      const reporter = begin([a, b]);

      reporter.onTestEnd(a, attempt("passed", 0));
      reporter.onTestEnd(b, attempt("failed", 0, "boom"));
      reporter.onEnd({ status: "failed" });

      expect(marker(".tests-done")).toBe("failed");
      expect(marker(".all-passed")).toBeNull();
      expect(results()).toMatchObject({ completed: 2, total: 2 });
    });

    it("reports an all-green run as passed", () => {
      const [a, b] = [testCase("a", "a", 0), testCase("b", "b", 0)];
      const reporter = begin([a, b]);

      reporter.onTestEnd(a, attempt("passed", 0));
      reporter.onTestEnd(b, attempt("passed", 0));
      reporter.onEnd({ status: "passed" });

      expect(marker(".tests-done")).toBe("passed");
      expect(marker(".all-passed")).toBe("1");
    });
  });
});
