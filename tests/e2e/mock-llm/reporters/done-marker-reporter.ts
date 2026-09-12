/**
 * Custom Playwright reporter that writes marker files for CI coordination.
 *
 * Marker files are written to `.mock-llm-markers/` at the project root —
 * intentionally outside Playwright's `outputDir` (`test-results-mock-llm/`)
 * to avoid being cleaned up.
 *
 * Written markers:
 *   .results.json — written after EVERY test attempt; always has the latest
 *                   results so that even a mid-suite kill leaves usable data
 *   .tests-done   — written only once the run is over; content is
 *                   "passed" or "failed"
 *   .all-passed   — written only when the run finished and every case passed
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
 * `.tests-done` / `.all-passed` are written only from `onEnd()`, letting the
 * CI wrapper distinguish "still running" from "done".
 *
 * `onEnd()` is the only safe place to decide the run is over. `onTestEnd()`
 * fires once per *attempt*, and no property of a single attempt tells you
 * whether Playwright will schedule another one: under
 * `test.describe.configure({ mode: "serial" })` — which every mock-LLM spec
 * uses — a failure re-runs the whole group, so cases that already reported
 * `passed`, and cases cut short with `skipped`, run a second time. Counting
 * attempts, or trying to classify an attempt as final, declares the suite
 * complete while cases are still queued; the workflow's poll loop then breaks
 * on the marker and kills a run that had work left.
 */
class DoneMarkerReporter implements Reporter {
  private totalTests = 0;
  /**
   * Latest attempt per test *case*, keyed by `TestCase.id`, in first-seen
   * order. A retried case is one case, and its last attempt decides its
   * verdict — the same rule the report script applies when it reads
   * `lastResult` from Playwright's own JSON.
   */
  private testsById = new Map<string, TestRecord>();
  private runEnded = false;
  private markerDirCreated = false;

  onBegin(_config: unknown, suite: { allTests(): TestCase[] }) {
    this.totalTests = suite.allTests().length;
  }

  onTestEnd(test: TestCase, result: TestResult) {
    this.testsById.set(test.id, {
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
  }

  onEnd(_result: FullResult) {
    this.runEnded = true;
    this.writeResults();
    this.writeCompletionMarkers();
  }

  /**
   * The run is over and every declared case reported at least once.
   *
   * A run cut short — by the global timeout, `maxFailures`, a webServer
   * failure, or an interrupt — leaves fewer cases than declared and stays
   * `in_progress` in `.results.json`.
   */
  private isComplete() {
    return this.runEnded && this.testsById.size >= this.totalTests;
  }

  /**
   * Whether every case that reported ended in a passing state.
   *
   * Derived from the recorded cases rather than accumulated as attempts
   * arrive, so a case that fails and then passes on retry reads as a pass —
   * the verdict Playwright itself reports for a flaky test.
   *
   * Deliberately does not require every declared case to have reported: a run
   * cut short by the global timeout keeps the verdict it has today. Whether
   * such a run should be green is a separate question from the retry
   * arithmetic, and is tracked on its own.
   *
   * A suite that declared no tests, or one where nothing reported at all, is a
   * configuration failure rather than a pass.
   */
  private allPassed() {
    if (!this.runEnded || this.totalTests === 0 || this.testsById.size === 0) {
      return false;
    }
    for (const record of this.testsById.values()) {
      if (record.status !== "passed" && record.status !== "skipped") {
        return false;
      }
    }
    return true;
  }

  /** Flush per-test timing/error data — called after every test attempt. */
  private writeResults() {
    const status = this.isComplete()
      ? this.allPassed()
        ? "passed"
        : "failed"
      : "in_progress";
    try {
      this.ensureMarkerDir();
      writeFileSync(
        join(MARKER_DIR, ".results.json"),
        JSON.stringify({
          status,
          // One entry per case, so this always matches `tests.length` and can
          // never exceed `total` — the report script derives "N not run" from
          // `total - completed`.
          completed: this.testsById.size,
          total: this.totalTests,
          tests: [...this.testsById.values()],
        }),
      );
    } catch {
      // Don't crash Playwright if marker write fails
    }
  }

  /** Write .tests-done and .all-passed — only once the run is over. */
  private writeCompletionMarkers() {
    const passed = this.allPassed();
    try {
      this.ensureMarkerDir();
      writeFileSync(
        join(MARKER_DIR, ".tests-done"),
        passed ? "passed" : "failed",
      );
      if (passed) {
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
