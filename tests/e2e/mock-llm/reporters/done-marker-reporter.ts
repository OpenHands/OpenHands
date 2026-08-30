/**
 * Custom Playwright reporter that writes marker files for CI coordination.
 *
 * Marker files are written to `.mock-llm-markers/` at the project root —
 * intentionally outside Playwright's `outputDir` (`test-results-mock-llm/`)
 * to avoid being cleaned up.
 *
 * Written markers:
 *   .results.json — written after EVERY test; always has the latest results
 *                   so that even a mid-suite kill leaves usable data
 *   .tests-done   — written only when all tests complete; content is
 *                   "passed" or "failed"
 *   .all-passed   — written only when all tests passed
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type {
  FullResult,
  Reporter,
  TestCase,
  TestResult,
} from "@playwright/test/reporter";

// Playwright runs from the project root (where the config file lives).
const MARKER_DIR = join(process.cwd(), ".mock-llm-markers");

interface TestRecord {
  title: string;
  status: string;
  durationMs: number;
  error: string;
}

/**
 * Tracks test results and writes them incrementally.
 *
 * `.results.json` is flushed after every `onTestEnd()` so the CI report
 * script always has data — even when the process is killed mid-suite
 * (e.g. the CI polling deadline expires before all tests finish).
 *
 * `.tests-done` / `.all-passed` are written only when the full suite
 * completes, letting the CI wrapper distinguish "still running" from
 * "done".
 */
class DoneMarkerReporter implements Reporter {
  private totalTests = 0;
  private completedTests = 0;
  private allPassed = true;
  private tests: TestRecord[] = [];
  private markerDirCreated = false;
  /** Test cases already counted toward `completedTests` (keyed by test.id). */
  private countedTestCases = new Set<string>();

  onBegin(_config: unknown, suite: { allTests(): TestCase[] }) {
    this.totalTests = suite.allTests().length;
  }

  onTestEnd(test: TestCase, result: TestResult) {
    // onTestEnd fires once per ATTEMPT. A failed attempt that still has
    // retries left is not the final state of this test case, so skip it.
    // Otherwise completedTests counts attempts while totalTests counts
    // cases, the completion marker fires before the suite ends, and CI
    // kills tests that were still queued. Mirrors Playwright's willRetry().
    const willRetry =
      test.outcome() === "unexpected" && test.results.length <= test.retries;
    if (willRetry) {
      return;
    }

    if (!this.countedTestCases.has(test.id)) {
      this.completedTests++;
      this.countedTestCases.add(test.id);
    }

    // test.outcome() reflects the final verdict across attempts:
    // "expected" (passed), "flaky" (failed then passed on retry) and
    // "skipped" are green; "unexpected" (failed every attempt) is red.
    const outcome = test.outcome();
    const passed =
      outcome === "expected" || outcome === "flaky" || outcome === "skipped";
    if (!passed) {
      this.allPassed = false;
    }

    this.tests.push({
      title: test.titlePath().filter(Boolean).join(" › "),
      status: result.status,
      durationMs: result.duration,
      error: result.errors
        .map((e) => e.message ?? "")
        .filter(Boolean)
        .join("\n\n")
        .slice(0, 1500),
    });

    // Always flush results so a mid-suite kill still leaves usable data.
    this.writeResults();

    // Write completion markers only after the last test case.
    if (this.completedTests >= this.totalTests) {
      this.writeCompletionMarkers();
    }
  }

  onEnd(_result: FullResult) {
    // Fallback: if onTestEnd never fired (webServer timeout, config
    // error, etc.), treat that as a failure and write what we have.
    if (this.totalTests === 0 || this.completedTests === 0) {
      this.allPassed = false;
    }
    this.writeResults();
    this.writeCompletionMarkers();
  }

  /** Flush per-test timing/error data — called after every test. */
  private writeResults() {
    const done = this.completedTests >= this.totalTests;
    const status = done
      ? this.allPassed
        ? "passed"
        : "failed"
      : "in_progress";
    try {
      this.ensureMarkerDir();
      writeFileSync(
        join(MARKER_DIR, ".results.json"),
        JSON.stringify({
          status,
          completed: this.completedTests,
          total: this.totalTests,
          tests: this.tests,
        }),
      );
    } catch {
      // Don't crash Playwright if marker write fails
    }
  }

  /** Write .tests-done and .all-passed — only when the suite is complete. */
  private writeCompletionMarkers() {
    const status = this.allPassed ? "passed" : "failed";
    try {
      this.ensureMarkerDir();
      writeFileSync(join(MARKER_DIR, ".tests-done"), status);
      if (this.allPassed) {
        writeFileSync(join(MARKER_DIR, ".all-passed"), "1");
      }
    } catch {
      // Don't crash Playwright if marker write fails
    }
  }

  private ensureMarkerDir() {
    if (!this.markerDirCreated) {
      mkdirSync(MARKER_DIR, { recursive: true });
      this.markerDirCreated = true;
    }
  }
}

export default DoneMarkerReporter;
