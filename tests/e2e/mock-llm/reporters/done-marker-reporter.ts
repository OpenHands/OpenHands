/**
 * Incremental mock-LLM results and completion markers for the CI wrappers.
 * Keep these outside Playwright's outputDir, which it clears during startup.
 */
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { stripVTControlCharacters } from "node:util";
import type {
  FullResult,
  Reporter,
  TestCase,
  TestError,
  TestResult,
} from "@playwright/test/reporter";

const MARKER_DIR = join(process.cwd(), ".mock-llm-markers");

interface TestRecord {
  title: string;
  status: string;
  durationMs: number;
  error: string;
}

let finishActiveRun: (() => void) | undefined;

/** Global teardown runs after test scheduling, before webServer teardown. */
export function finishTestRun() {
  finishActiveRun?.();
}

class DoneMarkerReporter implements Reporter {
  private totalTests = 0;
  private testsById = new Map<string, TestRecord>();
  private schedulingFinished = false;
  private hasRunError = false;

  onBegin(_config: unknown, suite: { allTests(): TestCase[] }) {
    finishActiveRun = () => this.finish();
    this.totalTests = suite.allTests().length;
  }

  onTestEnd(test: TestCase, result: TestResult) {
    // Expected failures pass; an unexpected pass fails. Skips are intentional
    // or follow a serial failure, whose final record keeps the suite red.
    let status: string = result.status;
    if (status !== "skipped") {
      if (result.status === test.expectedStatus) status = "passed";
      else if (status === "passed") status = "failed";
    }
    this.testsById.set(test.id, {
      title: test.titlePath().filter(Boolean).join(" › "),
      status,
      durationMs: result.duration,
      error:
        status === "passed"
          ? ""
          : result.errors
              .map((error) => error.message ?? "")
              .filter(Boolean)
              .join("\n\n")
              .slice(0, 1500),
    });
    // A serial retry reruns passed/skipped cases too: even seeing every case
    // once does not mean scheduling is over. Publish only incremental results.
    this.writeResults();
  }

  onError(error: TestError) {
    // The pinned Playwright runner reports a webServer teardown timeout via
    // onError too. Preserve the workflow's grace only after scheduling ended;
    // all other global/worker errors, including late cleanup failures, fail.
    const teardownTimeout =
      this.schedulingFinished &&
      !error.stack &&
      /^Timed out waiting [\d.]+s for the teardown for plugin setup to run$/.test(
        stripVTControlCharacters(error.message ?? ""),
      );
    if (teardownTimeout) return;
    this.hasRunError = true;
    if (this.schedulingFinished) this.finish();
  }

  onEnd(result: FullResult) {
    // Without the pre-teardown signal, a non-passing result may reflect
    // interrupted scheduling. A genuine failure always invalidates markers.
    if (
      result.status === "failed" ||
      (!this.schedulingFinished && result.status !== "passed")
    ) {
      this.hasRunError = true;
    }
    this.finish();
  }

  private finish() {
    this.schedulingFinished = true;
    this.writeResults();
    const passed = this.allPassed();
    try {
      mkdirSync(MARKER_DIR, { recursive: true });
      const passMarker = join(MARKER_DIR, ".all-passed");
      if (passed) writeFileSync(passMarker, "1");
      else rmSync(passMarker, { force: true });
      // Publish the completion signal last, after its results and verdict.
      writeFileSync(
        join(MARKER_DIR, ".tests-done"),
        passed ? "passed" : "failed",
      );
    } catch {
      // Reporting must not crash Playwright.
    }
  }

  private allPassed() {
    return (
      this.schedulingFinished &&
      !this.hasRunError &&
      this.totalTests > 0 &&
      this.testsById.size === this.totalTests &&
      [...this.testsById.values()].every(
        (test) => test.status === "passed" || test.status === "skipped",
      )
    );
  }

  private writeResults() {
    const complete =
      this.schedulingFinished && this.testsById.size === this.totalTests;
    try {
      mkdirSync(MARKER_DIR, { recursive: true });
      writeFileSync(
        join(MARKER_DIR, ".results.json"),
        JSON.stringify({
          status: complete
            ? this.allPassed()
              ? "passed"
              : "failed"
            : "in_progress",
          completed: this.testsById.size,
          total: this.totalTests,
          tests: [...this.testsById.values()],
        }),
      );
    } catch {
      // A mid-suite kill should retain the latest successfully written data.
    }
  }
}

export default DoneMarkerReporter;
