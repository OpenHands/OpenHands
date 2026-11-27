#!/usr/bin/env node

/**
 * Reads Playwright JSON results and renders a Markdown report for a PR comment.
 *
 * Usage:
 *   node render-mock-llm-report.mjs \
 *     --results test-results-mock-llm/results.json \
 *     --output  mock-llm-report.md \
 *     [--workflow-url <url>] \
 *     [--commit <sha>] \
 *     [--artifact-url <url>] \
 *     [--new-files <comma-separated spec paths added in this PR>]
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

// ── CLI args ───────────────────────────────────────────────────────────

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const [rawKey, inlineVal] = arg.slice(2).split("=", 2);
    const key = rawKey.replaceAll("-", "_");
    if (inlineVal !== undefined) {
      args[key] = inlineVal;
    } else {
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        args[key] = next;
        i++;
      } else {
        args[key] = "";
      }
    }
  }
  return args;
}

// ── Playwright JSON parsing ────────────────────────────────────────────

function loadResults(path) {
  if (!path || !existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
}

function collectTests(suites, parents = [], parentFile = "") {
  const tests = [];
  for (const suite of suites ?? []) {
    const titles = [...parents, suite.title].filter(Boolean);
    // Playwright's JSON reporter sets `file` on each suite/spec
    const suiteFile = suite.file || parentFile;
    for (const spec of suite.specs ?? []) {
      for (const test of spec.tests ?? []) {
        const results = test.results ?? [];
        const lastResult = results.at(-1);
        const duration = results.reduce(
          (sum, r) => sum + (Number(r.duration) || 0),
          0,
        );
        tests.push({
          title: [...titles, spec.title].filter(Boolean).join(" › "),
          file: spec.file || suiteFile,
          status: lastResult?.status ?? (spec.ok ? "passed" : "unknown"),
          durationMs: duration,
          retryCount: Math.max(0, results.length - 1),
          error: extractError(lastResult),
        });
      }
    }
    tests.push(...collectTests(suite.suites, titles, suiteFile));
  }
  return tests;
}

function extractError(result) {
  if (!result) return "";
  const errorMessages = Array.isArray(result.errors)
    ? result.errors
        .map((e) => e.message)
        .filter(Boolean)
        .join("\n\n")
    : "";
  const msg = result.error?.message ?? errorMessages;
  // Trim to avoid bloating the comment
  return sanitize(msg).slice(0, 1500);
}

function sanitize(str) {
  if (!str) return "";
  // Strip ANSI escape codes
  return str.replace(/\x1B\[[0-9;]*m/g, "");
}

// ── Formatting ─────────────────────────────────────────────────────────

function statusIcon(status) {
  switch (status) {
    case "passed":
      return "✅";
    case "failed":
      return "❌";
    case "timedOut":
      return "⏱️";
    case "skipped":
      return "⏭️";
    case "interrupted":
      return "🛑";
    default:
      return "❓";
  }
}

function formatDuration(ms) {
  if (ms < 1000) return `${ms}ms`;
  const secs = (ms / 1000).toFixed(1);
  return `${secs}s`;
}

function overallStatus(tests) {
  if (tests.length === 0) return "no tests";
  if (tests.every((t) => t.status === "passed")) return "passed";
  if (tests.some((t) => t.status === "failed" || t.status === "timedOut"))
    return "failed";
  return "mixed";
}

function overallIcon(status) {
  switch (status) {
    case "passed":
      return "✅";
    case "failed":
      return "❌";
    case "no tests":
      return "⚠️";
    default:
      return "🔶";
  }
}

// ── Report rendering ───────────────────────────────────────────────────

export function renderReport({
  tests,
  workflowUrl,
  commit,
  artifactUrl,
  title,
  newFiles,
  markerMeta,
}) {
  const status = overallStatus(tests);
  const icon = overallIcon(status);
  const passed = tests.filter((t) => t.status === "passed").length;
  const failed = tests.filter(
    (t) => t.status === "failed" || t.status === "timedOut",
  ).length;
  const skipped = tests.filter((t) => t.status === "skipped").length;
  const total = tests.length;
  const wasKilledMidSuite =
    markerMeta?.status === "in_progress" &&
    markerMeta.total > markerMeta.completed;

  // Determine which tests are new (from newly added spec files).
  // Playwright's JSON file paths are relative to testDir (e.g. "mock-llm-skills.spec.ts")
  // while --new-files paths are repo-relative (e.g. "tests/e2e/mock-llm/mock-llm-skills.spec.ts").
  // Match by basename or suffix in either direction.
  const newFileSet = new Set(newFiles ?? []);
  const basename = (p) => p.split("/").pop();
  const isNewTest = (t) =>
    newFileSet.size > 0 &&
    t.file &&
    [...newFileSet].some(
      (nf) =>
        t.file === nf ||
        basename(t.file) === basename(nf) ||
        nf.endsWith(`/${t.file}`) ||
        t.file.endsWith(`/${nf}`),
    );
  const newCount = tests.filter(isNewTest).length;

  const lines = [];

  // Header — use 🛑 when killed mid-suite so it's visually distinct
  const headerIcon = wasKilledMidSuite ? "🛑" : icon;
  lines.push(`## ${headerIcon} ${title || "Mock-LLM E2E Tests"}`);
  lines.push("");

  // Summary line
  const parts = [`**${passed}/${total} passed**`];
  if (failed) parts.push(`**${failed} failed**`);
  if (skipped) parts.push(`${skipped} skipped`);
  if (newCount) parts.push(`🆕 ${newCount} new`);
  if (wasKilledMidSuite) {
    const notRun = markerMeta.total - markerMeta.completed;
    parts.push(
      `⚠️ **${notRun} not run** (process killed at ${markerMeta.completed}/${markerMeta.total})`,
    );
  }
  lines.push(parts.join(" · "));
  lines.push("");

  // Metadata
  const meta = [];
  if (commit) meta.push(`Commit: \`${commit.slice(0, 8)}\``);
  if (workflowUrl) meta.push(`[Workflow run](${workflowUrl})`);
  if (artifactUrl) meta.push(`[Test artifacts](${artifactUrl})`);
  if (meta.length) {
    lines.push(meta.join(" · "));
    lines.push("");
  }

  lines.push("<details>");
  lines.push("<summary>Details</summary>");
  lines.push("");

  // New-tests callout (prominent, above the table)
  if (newCount > 0) {
    const newTests = tests.filter(isNewTest);
    // Group new tests by spec file
    const byFile = new Map();
    for (const t of newTests) {
      const key = t.file || "unknown";
      if (!byFile.has(key)) byFile.set(key, []);
      byFile.get(key).push(t);
    }
    lines.push(
      `> **🟢 ${newCount} new test${newCount === 1 ? "" : "s"} added in this PR**`,
    );
    for (const [file, fileTests] of byFile) {
      for (const t of fileTests) {
        lines.push(
          `> - ${statusIcon(t.status)} \`${file}\` › ${t.title.replace(/^.*› /, "")}`,
        );
      }
    }
    lines.push("");
  }

  // Test results table
  lines.push("| Status | Test | Duration |");
  lines.push("|:------:|------|----------|");
  for (const t of tests) {
    const retryNote = t.retryCount > 0 ? ` (${t.retryCount} retries)` : "";
    lines.push(
      `| ${statusIcon(t.status)} | ${t.title}${retryNote} | ${formatDuration(t.durationMs)} |`,
    );
  }
  lines.push("");

  // Error details for failed tests
  const failures = tests.filter(
    (t) => (t.status === "failed" || t.status === "timedOut") && t.error,
  );
  if (failures.length > 0) {
    lines.push("<details>");
    lines.push(`<summary>🔍 Failure details (${failures.length})</summary>`);
    lines.push("");
    for (const t of failures) {
      lines.push(`### ${statusIcon(t.status)} ${t.title}`);
      lines.push("");
      lines.push("```");
      lines.push(t.error);
      lines.push("```");
      lines.push("");
    }
    lines.push("</details>");
    lines.push("");
  }

  lines.push(
    "<sub>Posted by the Mock-LLM E2E workflow · results are deterministic (scripted LLM responses)</sub>",
  );
  lines.push("");
  lines.push("</details>");
  lines.push("");

  return lines.join("\n");
}

// ── Main ───────────────────────────────────────────────────────────────

function main() {
  const args = parseArgs(process.argv.slice(2));
  const resultsPath = args.results || "test-results-mock-llm/results.json";
  const outputPath = args.output || "mock-llm-report.md";

  const data = loadResults(resultsPath);
  let tests = data ? collectTests(data.suites) : [];

  // When Playwright is killed during webServer teardown (or mid-suite),
  // the JSON reporter never flushes results.json. Fall back to .results.json
  // written incrementally by DoneMarkerReporter after every onTestEnd().
  let markerMeta = null;
  if (!data || tests.length === 0) {
    const markerDir = args.marker_dir || ".mock-llm-markers";
    const markerResultsPath = `${markerDir}/.results.json`;
    const donePath = `${markerDir}/.tests-done`;

    if (existsSync(markerResultsPath)) {
      // Rich results from DoneMarkerReporter — has per-test timing & errors.
      // May be partial (status: "in_progress") if the process was killed
      // before all tests finished.
      const markerData = JSON.parse(readFileSync(markerResultsPath, "utf8"));
      tests = (markerData.tests ?? []).map((t) => ({
        title: t.title,
        status: t.status,
        durationMs: t.durationMs ?? 0,
        retryCount: 0,
        error: t.error ?? "",
      }));
      markerMeta = {
        status: markerData.status,
        completed: markerData.completed ?? tests.length,
        total: markerData.total ?? tests.length,
      };
      console.log(
        `No results.json; using marker results (${tests.length} tests run, ${markerMeta.completed}/${markerMeta.total} completed, status: ${markerData.status})`,
      );
    } else if (existsSync(donePath)) {
      // Minimal fallback — just pass/fail status, no timing
      const markerStatus = readFileSync(donePath, "utf8").trim();
      console.log(
        `No results.json; using done marker (status: ${markerStatus})`,
      );
      tests = [
        {
          title: "mock-LLM agent-server conversation",
          status: markerStatus === "passed" ? "passed" : "failed",
          durationMs: 0,
          retryCount: 0,
          error:
            markerStatus !== "passed"
              ? "Test failed (details in workflow logs)"
              : "",
        },
      ];
    } else {
      // No results file AND no marker files — Playwright was likely killed
      // before the DoneMarkerReporter could run. Check the exit code to
      // distinguish a genuine timeout from other failures.
      const exitCode = args.exit_code || "";
      if (exitCode === "124") {
        console.warn(
          `Warning: test suite timed out (exit code 124) — no results were collected`,
        );
        tests = [
          {
            title: "(test suite timed out before completing)",
            status: "timedOut",
            durationMs: 0,
            retryCount: 0,
            error:
              "The CI wrapper killed the Playwright process after the 5-minute deadline. " +
              "No test results were collected. Check the workflow logs for details.",
          },
        ];
      } else {
        console.warn(
          `Warning: no results file at ${resultsPath} and no marker files` +
            (exitCode ? ` (exit code: ${exitCode})` : ""),
        );
      }
    }
  }

  // Parse --new-files: comma-separated list of spec file paths added in this PR
  const newFiles = args.new_files
    ? args.new_files
        .split(",")
        .map((f) => f.trim())
        .filter(Boolean)
    : [];

  const report = renderReport({
    tests,
    workflowUrl: args.workflow_url || "",
    commit: args.commit || "",
    artifactUrl: args.artifact_url || "",
    title: args.title || "",
    newFiles,
    markerMeta,
  });

  writeFileSync(outputPath, report);
  console.log(
    `Report written to ${outputPath} (${tests.length} tests, ${overallStatus(tests)})`,
  );
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main();
}
