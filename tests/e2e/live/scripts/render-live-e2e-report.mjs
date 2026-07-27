#!/usr/bin/env node

import { existsSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve } from "node:path";

const DEFAULT_MARKER = "<!-- agent-canvas-live-e2e-report -->";

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) {
      continue;
    }

    const [rawKey, inlineValue] = arg.slice(2).split("=", 2);
    const key = rawKey.replaceAll("-", "_");
    if (inlineValue !== undefined) {
      args[key] = inlineValue;
      continue;
    }

    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      args[key] = next;
      index += 1;
    } else {
      args[key] = "";
    }
  }
  return args;
}

function assertWithinCwd(label, path) {
  const relativePath = relative(realpathSync(resolve(".")), path);
  if (relativePath.startsWith("..") || isAbsolute(relativePath)) {
    throw new Error(`${label} must be within the current working directory.`);
  }
}

function nearestExistingPath(path) {
  let currentPath = path;
  while (!existsSync(currentPath)) {
    const parentPath = dirname(currentPath);
    if (parentPath === currentPath) {
      throw new Error(`No existing parent found for ${path}`);
    }
    currentPath = parentPath;
  }
  return currentPath;
}

function resolveWithinCwd(label, filePath, options = {}) {
  const resolvedPath = resolve(filePath);
  if (existsSync(resolvedPath)) {
    assertWithinCwd(label, realpathSync(resolvedPath));
    return resolvedPath;
  }

  if (options.mustExist) {
    throw new Error(`${label} does not exist: ${resolvedPath}`);
  }

  const existingParent = nearestExistingPath(dirname(resolvedPath));
  const checkedPath = resolve(
    realpathSync(existingParent),
    relative(existingParent, resolvedPath),
  );
  assertWithinCwd(label, checkedPath);
  return resolvedPath;
}

function readJson(path) {
  if (!path) {
    return null;
  }

  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function collectSpecs(suites, parents = []) {
  const specs = [];
  for (const suite of suites ?? []) {
    const titles = [...parents, suite.title].filter(Boolean);
    for (const spec of suite.specs ?? []) {
      for (const test of spec.tests ?? []) {
        const results = test.results ?? [];
        const lastResult = results.at(-1);
        const duration = results.reduce(
          (total, result) => total + (Number(result.duration) || 0),
          0,
        );
        specs.push({
          title: [...titles, spec.title].filter(Boolean).join(" > "),
          project: test.projectName || "",
          status: lastResult?.status ?? (spec.ok ? "passed" : "unknown"),
          durationMs: duration,
          retryCount: Math.max(0, results.length - 1),
          attachments: collectAttachments(results),
          error: formatError(lastResult),
        });
      }
    }
    specs.push(...collectSpecs(suite.suites, titles));
  }
  return specs;
}

function collectAttachments(results) {
  return results.flatMap((result) =>
    (result.attachments ?? [])
      .filter((attachment) => attachment.path || attachment.name)
      .map((attachment) => ({
        name: attachment.name || "attachment",
        contentType: attachment.contentType || "",
        path: displayAttachmentPath(attachment.path || ""),
      })),
  );
}

function formatError(result) {
  const errorMessages = Array.isArray(result?.errors)
    ? result.errors
        .map((error) => error.message)
        .filter(Boolean)
        .join("\n\n")
    : "";
  const message = result?.error?.message ?? errorMessages;
  return sanitizeForComment(message).slice(0, 2000);
}

function sanitizeForComment(value) {
  return stripAnsi(value)
    .replaceAll("@OpenHands", "@\u200BOpenHands")
    .replaceAll("@openhands", "@\u200Bopenhands");
}

function stripAnsi(value) {
  return String(value).replace(/\u001B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, "");
}

function escapeCell(value) {
  return sanitizeForComment(value)
    .replaceAll("|", "\\|")
    .replaceAll("\n", "<br>");
}

function escapeCodeCell(value) {
  return escapeCell(value).replaceAll("`", "\\`");
}

function displayAttachmentPath(path) {
  const normalizedPath = sanitizeForComment(path).replaceAll("\\", "/");
  for (const anchor of ["test-results-live/", "playwright-report-live/"]) {
    const index = normalizedPath.indexOf(anchor);
    if (index >= 0) {
      return normalizedPath.slice(index);
    }
  }
  return normalizedPath;
}

function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms <= 0) {
    return "--";
  }

  const seconds = ms / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }

  const rounded = Math.round(seconds);
  const minutes = Math.floor(rounded / 60);
  const remainingSeconds = rounded % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

function formatStatus(status) {
  switch (status) {
    case "running":
      return "Running";
    case "passed":
      return "Passed";
    case "failed":
      return "Failed";
    case "skipped":
      return "Skipped";
    case "timedOut":
      return "Timed out";
    default:
      return status ? status[0].toUpperCase() + status.slice(1) : "Unknown";
  }
}

function inferOverallStatus(requestedStatus, stats) {
  if (requestedStatus && requestedStatus !== "auto") {
    return requestedStatus;
  }
  if (!stats) {
    return "unknown";
  }
  if ((stats.unexpected ?? 0) > 0 || (stats.interrupted ?? 0) > 0) {
    return "failed";
  }
  if ((stats.expected ?? 0) > 0 || (stats.flaky ?? 0) > 0) {
    return "passed";
  }
  return "skipped";
}

function buildSummary(results, status, reason) {
  const stats = results?.stats;
  const specs = collectSpecs(results?.suites);
  const passed = specs.filter((spec) => spec.status === "passed").length;
  const failed = specs.filter((spec) =>
    ["failed", "timedOut", "interrupted"].includes(spec.status),
  ).length;
  const skipped = specs.filter((spec) => spec.status === "skipped").length;
  const durationMs =
    Number(stats?.duration) ||
    specs.reduce((total, spec) => total + spec.durationMs, 0);

  return {
    status,
    reason,
    specs,
    passed,
    failed,
    skipped,
    total: specs.length,
    durationMs,
  };
}

function metadataLines(args) {
  const lines = [];
  if (args.model) {
    lines.push(`- Model: \`${sanitizeForComment(args.model)}\``);
  }
  if (args.commit) {
    lines.push(`- Commit: \`${sanitizeForComment(args.commit)}\``);
  }
  if (args.workflow_url) {
    lines.push(
      `- Workflow run: [open run](${sanitizeForComment(args.workflow_url)})`,
    );
  }
  if (args.artifact_url) {
    lines.push(
      `- Artifacts: [Playwright report, videos, screenshots](${sanitizeForComment(args.artifact_url)})`,
    );
  }
  if (args.timestamp) {
    lines.push(`- Generated: ${sanitizeForComment(args.timestamp)}`);
  }
  return lines;
}

function testTable(specs) {
  const lines = [
    "| Test | Project | Status | Duration | Retries |",
    "|------|---------|--------|----------|---------|",
  ];
  for (const spec of specs) {
    lines.push(
      `| ${escapeCell(spec.title)} | ${escapeCell(spec.project)} | ${formatStatus(
        spec.status,
      )} | ${formatDuration(spec.durationMs)} | ${spec.retryCount} |`,
    );
  }
  if (specs.length === 0) {
    lines.push(
      "| _No Playwright test result JSON found_ | -- | -- | -- | -- |",
    );
  }
  return lines;
}

function failureDetails(specs) {
  const failures = specs.filter((spec) => spec.error);
  if (failures.length === 0) {
    return [];
  }

  const lines = ["", "### Failures", ""];
  for (const failure of failures) {
    lines.push(`<details><summary>${escapeCell(failure.title)}</summary>`);
    lines.push("");
    lines.push("```text");
    lines.push(failure.error);
    lines.push("```");
    lines.push("");
    lines.push("</details>");
    lines.push("");
  }
  return lines;
}

function attachmentRows(specs) {
  return specs.flatMap((spec) =>
    (spec.attachments ?? []).map((attachment) => ({
      test: spec.title,
      ...attachment,
    })),
  );
}

function isVideoAttachment(attachment) {
  return (
    attachment.contentType.startsWith("video/") ||
    /\.(mp4|mov|webm)$/i.test(attachment.path)
  );
}

function isImageAttachment(attachment) {
  return (
    attachment.contentType.startsWith("image/") ||
    /\.(gif|jpe?g|png|svg)$/i.test(attachment.path)
  );
}

function evidenceDetails(specs, args) {
  const attachments = attachmentRows(specs);
  const videos = attachments.filter(isVideoAttachment);
  const images = attachments.filter(isImageAttachment);

  if (
    attachments.length === 0 &&
    !args.artifact_url &&
    !args.video_url &&
    !args.video_preview_url &&
    !args.screenshot_url
  ) {
    return [];
  }

  const lines = [
    "",
    "<details>",
    "<summary>View Playwright video and artifacts</summary>",
    "",
  ];

  if (args.video_preview_url) {
    lines.push(
      "**Recorded video:**",
      "",
      `![Live Agent E2E recording](${sanitizeForComment(args.video_preview_url)})`,
      "",
    );
    if (args.video_url) {
      lines.push(
        `[Open full WebM recording](${sanitizeForComment(args.video_url)})`,
        "",
      );
    }
  } else if (args.video_url) {
    lines.push(sanitizeForComment(args.video_url), "");
  } else if (videos.length > 0) {
    lines.push(`**Recorded video:** \`${videos[0].path}\``);
    if (args.artifact_url) {
      lines.push("");
      lines.push(
        "The Playwright video is inside the uploaded artifact. GitHub Actions artifacts are downloadable archives, so they are linked here instead of embedded inline.",
      );
    }
    lines.push("");
  }

  if (args.screenshot_url) {
    lines.push(
      `![Live Agent response](${sanitizeForComment(args.screenshot_url)})`,
      "",
    );
  } else if (images.length > 0) {
    lines.push(`**Screenshot:** \`${images[0].path}\``, "");
  }

  if (args.artifact_url) {
    lines.push(
      `- Full artifact: [Playwright report, videos, screenshots](${sanitizeForComment(args.artifact_url)})`,
    );
  }
  lines.push(
    "- HTML report path in artifact: `playwright-report-live/index.html`",
  );
  if (args.workflow_url) {
    lines.push(
      `- Workflow run: [open run](${sanitizeForComment(args.workflow_url)})`,
    );
  }
  lines.push("");

  if (
    attachments.length > 0 &&
    !args.video_url &&
    !args.video_preview_url &&
    !args.screenshot_url
  ) {
    lines.push(
      "| Test | Attachment | Type | Location |",
      "|------|------------|------|----------|",
    );
    for (const attachment of attachments) {
      lines.push(
        `| ${escapeCell(attachment.test)} | ${escapeCell(
          attachment.name,
        )} | ${escapeCell(attachment.contentType || "--")} | \`${escapeCodeCell(
          attachment.path || "--",
        )}\` |`,
      );
    }
    lines.push("");
  }

  lines.push("</details>");
  lines.push("");
  return lines;
}

function buildReport(args) {
  const resultsPath = args.results
    ? resolveWithinCwd("results", args.results)
    : null;
  const results = readJson(resultsPath);
  const status = inferOverallStatus(args.status, results?.stats);
  const summary = buildSummary(results, status, args.reason);

  const lines = [
    args.marker || DEFAULT_MARKER,
    "## Agent Canvas Live E2E",
    "",
    `**Status:** ${formatStatus(summary.status)}`,
  ];

  if (summary.reason) {
    lines.push(`**Reason:** ${sanitizeForComment(summary.reason)}`);
  }

  lines.push(
    `**Summary:** ${summary.passed}/${summary.total} passed, ${summary.failed} failed, ${summary.skipped} skipped, ${formatDuration(
      summary.durationMs,
    )} total`,
    "",
  );

  const metadata = metadataLines(args);
  if (metadata.length > 0) {
    lines.push(...metadata, "");
  }

  if (summary.status === "running") {
    lines.push(
      "The live Agent Server E2E run is in progress. This comment will be updated with the final result.",
    );
    lines.push("");
  }

  if (results?.error) {
    lines.push(
      `_Could not read Playwright JSON results: ${sanitizeForComment(results.error)}_`,
      "",
    );
  }

  lines.push(...testTable(summary.specs));
  lines.push(...evidenceDetails(summary.specs, args));
  lines.push(...failureDetails(summary.specs));

  return lines.join("\n").trimEnd() + "\n";
}

const args = parseArgs(process.argv.slice(2));
const report = buildReport(args);

if (args.output) {
  writeFileSync(resolveWithinCwd("output", args.output), report);
} else {
  process.stdout.write(report);
}
