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
 *
 * Counting rule (regression guard): when `retries` is configured,
 * Playwright fires `onTestEnd` once per *attempt*, not once per case.
 * Naive counting makes `completedTests` exceed `totalTests` mid-suite,
 * which writes the completion marker before the last case has run, and
 * also prevents `.all-passed` from being written because a transient
 * failure on the first attempt of a retried test clears `allPassed`
 * even when the retry passes. We only count the first attempt of each
 * case (`result.retry === 0`) toward completion and reset `allPassed`
 * in `onRetry` so a successful retry re-arms it.
 */
class DoneMarkerReporter implements Reporter {
  private totalTests = 0;
  private completedCases = 0;
  private failedCases = new Set<string>();
  private allPassed = true;
  private tests: TestRecord[] = [];
  private markerDirCreated = false;

  onBegin(_config: unknown, suite: { allTests(): TestCase[] }) {
    this.totalTests = suite.allTests().length;
  }

  /**
   * Fires before each retry. Reset `allPassed` so the upcoming attempt
   * can re-arm it; otherwise the first failure sticks across retries.
   */
  onRetry(_test: TestCase, _result: TestResult) {
    this.allPassed = true;
  }

  onTestEnd(test: TestCase, result: TestResult) {
    // Only count the first attempt of each case toward completion.
    // Retries fire onTestEnd again with result.retry > 0; ignore those.
    const isFirstAttempt = result.retry === 0;
    if (isFirstAttempt) {
      this.completedCases++;
    }
    const passed = result.status === "passed" || result.status === "skipped";
    if (!passed) {
      if (isFirstAttempt) {
        const caseKey = this.caseKey(test);
        this.failedCases.add(caseKey);
      }
      this.allPassed = false;
    } else if (isFirstAttempt) {
      // A successful first attempt short-circuits this case; mark it as
      // observed so a later retry of the same case does not double-count.
      const caseKey = this.caseKey(test);
      this.failedCases.delete(caseKey);
    }

    if (isFirstAttempt) {
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
    }

    // Always flush results so a mid-suite kill still leaves usable data.
    this.writeResults();

    // Write completion markers only after the last case has had its
    // first attempt. Retries may still be pending at this point, but
    // they are absorbed by the configured `retries` policy without
    // affecting the marker (the result is what Playwright reports).
    if (this.completedCases >= this.totalTests) {
      this.writeCompletionMarkers();
    }
  }

  onEnd(_result: FullResult) {
    // Fallback: if onTestEnd never fired (webServer timeout, config
    // error, etc.), treat that as a failure and write what we have.
    if (this.totalTests === 0 || this.completedCases === 0) {
      this.allPassed = false;
    }
    this.writeResults();
    this.writeCompletionMarkers();
  }

  /** Stable identity for a test case across retries. */
  private caseKey(test: TestCase): string {
    return test.titlePath().filter(Boolean).join(" › ");
  }

  /** Flush per-test timing/error data — called after every test. */
  private writeResults() {
    const done = this.completedCases >= this.totalTests;
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
          completed: this.completedCases,
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
