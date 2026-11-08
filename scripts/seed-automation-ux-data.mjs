#!/usr/bin/env node
/**
 * Seed local automation UX data (list, filters, detail, run history).
 *
 * Usage:
 *   node scripts/seed-automation-ux-data.mjs
 *
 * Env:
 *   AUTOMATION_BASE_URL  Ingress origin (default http://localhost:8100)
 *   SESSION_API_KEY      X-Session-API-Key (default ~/.openhands/agent-canvas/api-key.txt)
 *   AUTOMATION_DB        SQLite path (default .tmp/automation/automations.db)
 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");

const BASE_URL = (
  process.env.AUTOMATION_BASE_URL || "http://localhost:8100"
).replace(/\/$/, "");
const DB_PATH =
  process.env.AUTOMATION_DB ||
  join(repoRoot, ".tmp/automation/automations.db");
const API_KEY =
  process.env.SESSION_API_KEY ||
  readFileSync(
    join(homedir(), ".openhands/agent-canvas/api-key.txt"),
    "utf8",
  ).trim();

const hoursAgo = (hours) => new Date(Date.now() - hours * 3_600_000);

const SEEDS = [
  {
    name: "PR Triage Digest",
    prompt:
      "Review newly opened pull requests in acme/frontend-app, identify risky changes, summarize likely impact, and prepare a concise digest with priority ordering for the engineering review channel.",
    model: "triage-fast",
    timeout: 600,
    enabled: true,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: {
      type: "cron",
      schedule: "0 9 * * 1-5",
      timezone: "America/Los_Angeles",
    },
    scheduleHuman: "Weekdays at 09:00",
    lastTriggeredHoursAgo: 2,
    runs: [
      ["COMPLETED", 2, 0.42],
      ["COMPLETED", 26, 0.38],
      ["FAILED", 50, 0.11],
      ["COMPLETED", 74, 0.41],
      ["COMPLETED", 98, 0.36],
      ["COMPLETED", 170, 0.4],
      ["FAILED", 194, 0.09],
      ["COMPLETED", 218, 0.37],
      ["COMPLETED", 242, 0.39],
      ["COMPLETED", 266, 0.35],
    ],
  },
  {
    name: "Nightly Security Pass",
    prompt:
      "Scan the acme/backend-api repository for known security vulnerabilities, outdated dependencies, and insecure code patterns. Produce a prioritized remediation summary.",
    model: "security-careful",
    timeout: 900,
    enabled: true,
    repos: [{ url: "acme/backend-api", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "30 1 * * *", timezone: "UTC" },
    scheduleHuman: "Daily at 01:30",
    lastTriggeredHoursAgo: 8,
    runs: [
      ["COMPLETED", 8, 1.12],
      ["COMPLETED", 32, 1.05],
      ["COMPLETED", 56, 0.98],
      ["FAILED", 80, 0.22],
      ["COMPLETED", 104, 1.08],
    ],
  },
  {
    name: "Docs Sync on Push",
    prompt:
      "Monitor acme/docs for new pushes. For each push, generate a changelog-ready summary of what changed and why.",
    model: "docs-fast",
    enabled: true,
    repos: [{ url: "acme/docs", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "*/15 * * * *", timezone: "America/New_York" },
    scheduleHuman: "Every 15 minutes",
    lastTriggeredHoursAgo: 1,
    runs: [
      ["COMPLETED", 1, 0.08],
      ["CANCELLED", 18, null],
      ["SKIPPED", 42, null],
    ],
  },
  {
    name: "Release Readiness Review",
    prompt:
      "Compile a release readiness report: list open blockers, active incidents, and pending approvals for acme/realtime-service.",
    model: "release-review",
    enabled: false,
    repos: [{ url: "acme/realtime-service", ref: "release", provider: "github" }],
    trigger: { type: "cron", schedule: "0 11 * * 5", timezone: "America/Chicago" },
    scheduleHuman: "Fridays at 11:00",
    lastTriggeredHoursAgo: 14 * 24,
    runs: [
      ["FAILED", 14 * 24, null],
      ["COMPLETED", 21 * 24, 0.67],
    ],
  },
  {
    name: "Incident Webhook Summary",
    prompt:
      "Summarize incoming incident webhooks, categorize by severity, and post a digest to the on-call Slack channel.",
    model: "incident-summary",
    enabled: false,
    repos: [{ url: "acme/incident-service", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 */2 * * *", timezone: "UTC" },
    scheduleHuman: "Every 2 hours",
    lastTriggeredHoursAgo: null,
    runs: [],
  },
  {
    name: "PR Review on Open",
    prompt:
      "When a new PR is opened, perform a thorough code review focusing on correctness, security, and performance. Post findings as inline comments.",
    model: "review-fast",
    timeout: 1800,
    enabled: true,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: {
      type: "event",
      source: "github",
      on: "pull_request.opened",
      filter: "repository.full_name == 'acme/frontend-app'",
    },
    lastTriggeredHoursAgo: 3,
    runs: [
      ["COMPLETED", 3, 0.88],
      ["COMPLETED", 6, 0.79],
      ["FAILED", 20, 0.15],
      ["COMPLETED", 44, 0.81],
      ["COMPLETED", 68, 0.74],
    ],
  },
  {
    name: "Release Notes Generator",
    prompt:
      "Generate comprehensive release notes from the commits since the last release. Include breaking changes, new features, and bug fixes.",
    model: "docs-fast",
    enabled: true,
    repos: [{ url: "acme/backend-api", ref: "main", provider: "github" }],
    trigger: {
      type: "event",
      source: "github",
      on: "release.published",
      filter: "glob(release.tag_name, 'v*') && !release.prerelease",
    },
    lastTriggeredHoursAgo: 72,
    runs: [
      ["COMPLETED", 72, 0.54],
      ["COMPLETED", 240, 0.61],
    ],
  },
  {
    name: "Weekly Standup Digest",
    prompt:
      "Collect merged PRs, open incidents, and Linear tickets moved this week. Draft a standup digest for #eng-standup.",
    model: "standup-fast",
    enabled: true,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 9 * * 1", timezone: "America/Los_Angeles" },
    scheduleHuman: "Mondays at 09:00",
    lastTriggeredHoursAgo: 0.25,
    running: true,
    runs: [
      ["COMPLETED", 168, 0.29],
      ["COMPLETED", 336, 0.31],
      ["COMPLETED", 504, 0.27],
    ],
  },
  {
    name: "Slack Channel Monitor",
    prompt:
      "Watch #support for customer-reported regressions. When a thread looks like a product bug, file a Linear issue and link the Slack thread.",
    model: "support-fast",
    enabled: true,
    trigger: {
      type: "event",
      source: "slack",
      on: "message.channels",
      filter: "icontains(text, 'bug') || icontains(text, 'broken')",
    },
    lastTriggeredHoursAgo: 5,
    runs: [
      ["COMPLETED", 5, 0.19],
      ["COMPLETED", 12, 0.16],
      ["SKIPPED", 29, null],
      ["COMPLETED", 53, 0.21],
    ],
  },
  {
    name: "Dependabot Triage",
    prompt:
      "Review Dependabot PRs, group safe version bumps, and flag breaking major upgrades that need a human owner.",
    model: "triage-fast",
    enabled: true,
    repos: [{ url: "acme/backend-api", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 8 * * 1-5", timezone: "UTC" },
    scheduleHuman: "Weekdays at 08:00",
    lastTriggeredHoursAgo: 4,
    runs: [
      ["FAILED", 4, 0.07],
      ["FAILED", 28, 0.06],
      ["COMPLETED", 52, 0.33],
      ["FAILED", 76, 0.05],
    ],
  },
  {
    name: "Stale PR Nudge",
    prompt:
      "Find pull requests in acme/frontend-app that have had no review activity for 5 days. Post a polite nudge on the PR and summarize owners in #eng-reviews.",
    model: "triage-fast",
    enabled: true,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 10 * * 1-5", timezone: "America/Los_Angeles" },
    scheduleHuman: "Weekdays at 10:00",
    lastTriggeredHoursAgo: 6,
    runs: [
      ["COMPLETED", 6, 0.14],
      ["COMPLETED", 30, 0.12],
      ["COMPLETED", 54, 0.16],
      ["SKIPPED", 78, null],
    ],
  },
  {
    name: "Flaky Test Hunter",
    prompt:
      "Analyze the last 48 hours of CI on acme/frontend-app. Identify flaky tests, group by file, and open or update a tracking issue with reproduction hints.",
    model: "careful",
    timeout: 1200,
    enabled: true,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 7 * * *", timezone: "UTC" },
    scheduleHuman: "Daily at 07:00",
    lastTriggeredHoursAgo: 9,
    runs: [
      ["COMPLETED", 9, 1.44],
      ["FAILED", 33, 0.28],
      ["COMPLETED", 57, 1.31],
      ["COMPLETED", 81, 1.22],
    ],
  },
  {
    name: "License Compliance Sweep",
    prompt:
      "Scan acme/backend-api dependencies for GPL or unknown licenses. Produce a table of new findings since the last run.",
    model: "security-careful",
    enabled: true,
    repos: [{ url: "acme/backend-api", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 3 * * 1", timezone: "UTC" },
    scheduleHuman: "Mondays at 03:00",
    lastTriggeredHoursAgo: 20,
    runs: [
      ["COMPLETED", 20, 0.77],
      ["COMPLETED", 188, 0.81],
    ],
  },
  {
    name: "Changelog Drafter",
    prompt:
      "Draft this week's changelog for acme/docs from merged PRs labeled feature, fix, or breaking.",
    model: "docs-fast",
    enabled: true,
    repos: [{ url: "acme/docs", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 16 * * 5", timezone: "America/New_York" },
    scheduleHuman: "Fridays at 16:00",
    lastTriggeredHoursAgo: 48,
    runs: [
      ["COMPLETED", 48, 0.24],
      ["COMPLETED", 216, 0.22],
      ["CANCELLED", 384, null],
    ],
  },
  {
    name: "Broken Link Checker",
    prompt:
      "Crawl published docs in acme/docs, report broken internal and external links, and file issues for anything older than 7 days.",
    model: "docs-fast",
    enabled: false,
    repos: [{ url: "acme/docs", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 4 * * 0", timezone: "UTC" },
    scheduleHuman: "Sundays at 04:00",
    lastTriggeredHoursAgo: 36,
    runs: [
      ["FAILED", 36, null],
      ["COMPLETED", 204, 0.45],
    ],
  },
  {
    name: "Onboarding Buddy",
    prompt:
      "When a new engineer is added to the org, generate a first-week checklist from acme/handbook and post it to their Slack DM.",
    model: "standup-fast",
    enabled: true,
    repos: [{ url: "acme/handbook", ref: "main", provider: "github" }],
    trigger: {
      type: "event",
      source: "github",
      on: "membership.added",
      filter: "team.name == 'engineering'",
    },
    lastTriggeredHoursAgo: 96,
    runs: [
      ["COMPLETED", 96, 0.18],
    ],
  },
  {
    name: "Issue to Draft PR",
    prompt:
      "When a Linear issue is labeled 'ready-for-agent', create a draft PR in the linked repo with a first-pass implementation and a test plan.",
    model: "careful",
    timeout: 1800,
    enabled: true,
    repos: [{ url: "acme/backend-api", ref: "main", provider: "github" }],
    trigger: {
      type: "event",
      source: "linear",
      on: "issue.updated",
      filter: "contains(labels, 'ready-for-agent')",
    },
    lastTriggeredHoursAgo: 11,
    running: true,
    runs: [
      ["COMPLETED", 35, 2.18],
      ["FAILED", 59, 0.41],
      ["COMPLETED", 110, 1.96],
    ],
  },
  {
    name: "CI Failure Autopsy",
    prompt:
      "When a GitHub check suite fails on main, summarize the failing jobs, likely root cause, and whether this looks flaky or a real regression.",
    model: "careful",
    enabled: true,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: {
      type: "event",
      source: "github",
      on: "check_suite.completed",
      filter: "check_suite.conclusion == 'failure' && check_suite.head_branch == 'main'",
    },
    lastTriggeredHoursAgo: 1.5,
    runs: [
      ["COMPLETED", 1.5, 0.52],
      ["COMPLETED", 7, 0.48],
      ["FAILED", 14, 0.19],
      ["COMPLETED", 22, 0.55],
      ["COMPLETED", 31, 0.47],
    ],
  },
  {
    name: "Push Changelog Ping",
    prompt:
      "On every push to main in acme/docs, post a 3-bullet summary to #docs-updates.",
    model: "docs-fast",
    enabled: true,
    repos: [{ url: "acme/docs", ref: "main", provider: "github" }],
    trigger: {
      type: "event",
      source: "github",
      on: "push",
      filter: "ref == 'refs/heads/main'",
    },
    lastTriggeredHoursAgo: 0.8,
    runs: [
      ["COMPLETED", 0.8, 0.06],
      ["COMPLETED", 3.2, 0.05],
      ["SKIPPED", 5, null],
      ["COMPLETED", 9, 0.07],
      ["COMPLETED", 14, 0.06],
    ],
  },
  {
    name: "Review Comment Resolver",
    prompt:
      "When a reviewer leaves a comment containing '@openhands please fix', apply the requested change and reply with a summary of the edit.",
    model: "review-fast",
    timeout: 1500,
    enabled: true,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: {
      type: "event",
      source: "github",
      on: "pull_request_review_comment.created",
      filter: "icontains(comment.body, '@openhands please fix')",
    },
    lastTriggeredHoursAgo: 7,
    runs: [
      ["COMPLETED", 7, 0.63],
      ["COMPLETED", 19, 0.71],
      ["CANCELLED", 27, null],
    ],
  },
  {
    name: "Jira Bug to Repro",
    prompt:
      "When a Jira bug is moved to Ready, clone the linked repo, write a failing reproduction test, and attach the patch to the ticket.",
    model: "careful",
    enabled: false,
    repos: [{ url: "acme/realtime-service", ref: "main", provider: "github" }],
    trigger: {
      type: "event",
      source: "jira",
      on: "issue.updated",
      filter: "fields.status.name == 'Ready' && fields.issuetype.name == 'Bug'",
    },
    lastTriggeredHoursAgo: 60,
    runs: [
      ["FAILED", 60, 0.33],
    ],
  },
  {
    name: "Monthly Cost Report",
    prompt:
      "Summarize last month's LLM spend by automation, highlight outliers, and recommend timeouts or model changes.",
    model: "standup-fast",
    enabled: true,
    trigger: { type: "cron", schedule: "0 9 1 * *", timezone: "America/Los_Angeles" },
    scheduleHuman: "1st of the month at 09:00",
    lastTriggeredHoursAgo: 240,
    runs: [
      ["COMPLETED", 240, 0.31],
      ["COMPLETED", 960, 0.28],
    ],
  },
  {
    name: "Weekend On-call Brief",
    prompt:
      "Friday afternoon: compile open Sev-1/Sev-2 incidents, recent deploys, and a rollback cheat sheet for the weekend on-call.",
    model: "incident-summary",
    enabled: true,
    repos: [{ url: "acme/incident-service", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 16 * * 5", timezone: "America/Los_Angeles" },
    scheduleHuman: "Fridays at 16:00",
    lastTriggeredHoursAgo: 50,
    runs: [
      ["COMPLETED", 50, 0.39],
      ["COMPLETED", 218, 0.41],
      ["COMPLETED", 386, 0.36],
    ],
  },
  {
    name: "i18n Drift Check",
    prompt:
      "Compare src/i18n/translation.json keys against English source strings. List missing translations and unused keys.",
    model: "docs-fast",
    enabled: true,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "30 6 * * 1-5", timezone: "UTC" },
    scheduleHuman: "Weekdays at 06:30",
    lastTriggeredHoursAgo: 12,
    runs: [
      ["COMPLETED", 12, 0.17],
      ["COMPLETED", 36, 0.15],
      ["FAILED", 60, 0.04],
      ["COMPLETED", 84, 0.16],
    ],
  },
  {
    name: "Coverage Gate Watcher",
    prompt:
      "If a PR drops line coverage by more than 1%, comment with the files responsible and suggested tests.",
    model: "review-fast",
    enabled: true,
    repos: [{ url: "acme/backend-api", ref: "main", provider: "github" }],
    trigger: {
      type: "event",
      source: "github",
      on: "pull_request.synchronize",
      filter: "repository.full_name == 'acme/backend-api'",
    },
    lastTriggeredHoursAgo: 4.5,
    runs: [
      ["COMPLETED", 4.5, 0.27],
      ["SKIPPED", 8, null],
      ["COMPLETED", 16, 0.29],
      ["COMPLETED", 28, 0.25],
    ],
  },
  {
    name: "Draft PR Reminder",
    prompt:
      "Find draft PRs older than 10 days. Ask the author if they still intend to ship, and close with a comment if they agree.",
    model: "triage-fast",
    enabled: false,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 12 * * 3", timezone: "America/Los_Angeles" },
    scheduleHuman: "Wednesdays at 12:00",
    lastTriggeredHoursAgo: null,
    runs: [],
  },
  {
    name: "Design Token Audit",
    prompt:
      "Scan acme/frontend-app for hardcoded hex colors and spacing values that should use design tokens. Group by file and suggest replacements.",
    model: "docs-fast",
    enabled: true,
    repos: [{ url: "acme/frontend-app", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 5 * * 2", timezone: "UTC" },
    scheduleHuman: "Tuesdays at 05:00",
    lastTriggeredHoursAgo: 70,
    runs: [
      ["COMPLETED", 70, 0.58],
      ["COMPLETED", 238, 0.62],
    ],
  },
  {
    name: "Sentry Spike Explainer",
    prompt:
      "When Sentry error volume spikes 3x hour-over-hour, explain the top stack traces and whether a recent deploy is implicated.",
    model: "incident-summary",
    enabled: true,
    trigger: {
      type: "event",
      source: "sentry",
      on: "metric.alert",
      filter: "alert.name == 'error-volume-spike'",
    },
    lastTriggeredHoursAgo: 15,
    runs: [
      ["COMPLETED", 15, 0.44],
      ["FAILED", 40, 0.12],
      ["COMPLETED", 90, 0.49],
    ],
  },
  {
    name: "GitLab MR Review",
    prompt:
      "Review newly opened merge requests in acme/data-platform. Focus on SQL safety, partition filters, and cost of full-table scans.",
    model: "review-fast",
    timeout: 1800,
    enabled: true,
    repos: [{ url: "acme/data-platform", ref: "main", provider: "gitlab" }],
    trigger: {
      type: "event",
      source: "gitlab",
      on: "merge_request.opened",
    },
    lastTriggeredHoursAgo: 18,
    runs: [
      ["COMPLETED", 18, 0.91],
      ["COMPLETED", 41, 0.84],
      ["COMPLETED", 73, 0.88],
    ],
  },
  {
    name: "Bitbucket Nightly Diff",
    prompt:
      "Summarize commits landed in acme/legacy-billing since yesterday and flag schema migrations.",
    model: "triage-fast",
    enabled: true,
    repos: [{ url: "acme/legacy-billing", ref: "master", provider: "bitbucket" }],
    trigger: { type: "cron", schedule: "0 2 * * *", timezone: "UTC" },
    scheduleHuman: "Daily at 02:00",
    lastTriggeredHoursAgo: 13,
    runs: [
      ["COMPLETED", 13, 0.21],
      ["COMPLETED", 37, 0.19],
      ["FAILED", 61, null],
      ["COMPLETED", 85, 0.2],
    ],
  },
  {
    name: "Customer Quote Miner",
    prompt:
      "Read #win-stories and #support. Extract reusable customer quotes and file them in acme/handbook/sales-quotes.md.",
    model: "standup-fast",
    enabled: false,
    repos: [{ url: "acme/handbook", ref: "main", provider: "github" }],
    trigger: { type: "cron", schedule: "0 15 * * 5", timezone: "America/New_York" },
    scheduleHuman: "Fridays at 15:00",
    lastTriggeredHoursAgo: 400,
    runs: [
      ["COMPLETED", 400, 0.13],
    ],
  },
  {
    name: "Pending Dispatch Smoke",
    prompt:
      "No-op smoke automation used to preview a queued PENDING run in the activity log.",
    model: "fast",
    enabled: true,
    trigger: { type: "cron", schedule: "0 * * * *", timezone: "UTC" },
    scheduleHuman: "Hourly",
    lastTriggeredHoursAgo: 0.05,
    pending: true,
    runs: [
      ["COMPLETED", 1.1, 0.03],
      ["COMPLETED", 2.1, 0.03],
    ],
  },
];

function toHexUuid(id) {
  return id.replaceAll("-", "");
}

function sqlQuote(value) {
  if (value == null) return "NULL";
  if (typeof value === "number") return String(value);
  return `'${String(value).replaceAll("'", "''")}'`;
}

function sqlite(sql) {
  return execFileSync("sqlite3", [DB_PATH, sql], { encoding: "utf8" }).trim();
}

async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: {
      "X-Session-API-Key": API_KEY,
      "Content-Type": "application/json",
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!response.ok) {
    throw new Error(
      `${method} ${path} failed (${response.status}): ${typeof data === "string" ? data : JSON.stringify(data)}`,
    );
  }
  return data;
}

async function deleteExistingSeeds() {
  const list = await api("/api/automation/v1?limit=100");
  const names = new Set(["_probe", ...SEEDS.map((seed) => seed.name)]);
  for (const automation of list.automations ?? []) {
    if (!names.has(automation.name)) continue;
    await api(`/api/automation/v1/${automation.id}`, { method: "DELETE" });
    console.log(`deleted ${automation.name}`);
  }
}

function insertRuns(automationId, seed) {
  const hexAutomationId = toHexUuid(automationId);
  const rows = [];

  if (seed.running || seed.pending) {
    const started = hoursAgo(seed.pending ? 0.05 : 0.25);
    const timeout = new Date(Date.now() + 7 * 86_400_000);
    rows.push({
      id: toHexUuid(randomUUID()),
      status: seed.pending ? "PENDING" : "RUNNING",
      error: null,
      started,
      completed: null,
      conversationId: seed.pending
        ? null
        : `conv-ux-${toHexUuid(randomUUID()).slice(0, 8)}`,
      bashCommandId: seed.pending
        ? null
        : `cmd-ux-${toHexUuid(randomUUID()).slice(0, 8)}`,
      cost: null,
      timeout,
    });
  }

  for (const [status, hours, cost] of seed.runs) {
    const started = hoursAgo(hours);
    const durationMs =
      status === "SKIPPED" ? 8_000 : 90_000 + Math.round(Math.random() * 180_000);
    const completed =
      status === "RUNNING" ? null : new Date(started.getTime() + durationMs);
    const failedBeforeSandbox = status === "FAILED" && cost == null;
    rows.push({
      id: toHexUuid(randomUUID()),
      status,
      error:
        status === "FAILED"
          ? failedBeforeSandbox
            ? "Sandbox provisioning failed: no available runtime"
            : "Process exited with code 1"
          : null,
      started,
      completed,
      conversationId: failedBeforeSandbox
        ? null
        : `conv-ux-${toHexUuid(randomUUID()).slice(0, 8)}`,
      bashCommandId: failedBeforeSandbox
        ? null
        : `cmd-ux-${toHexUuid(randomUUID()).slice(0, 8)}`,
      cost,
      timeout: null,
    });
  }

  for (const row of rows) {
    sqlite(`
      INSERT INTO automation_runs (
        id, automation_id, status, error_detail, created_at, started_at,
        completed_at, conversation_id, timeout_at, sandbox_id, event_payload,
        bash_command_id, telemetry_distinct_id, cost
      ) VALUES (
        ${sqlQuote(row.id)},
        ${sqlQuote(hexAutomationId)},
        ${sqlQuote(row.status)},
        ${sqlQuote(row.error)},
        ${sqlQuote(row.started.toISOString())},
        ${sqlQuote(row.started.toISOString())},
        ${sqlQuote(row.completed ? row.completed.toISOString() : null)},
        ${sqlQuote(row.conversationId)},
        ${sqlQuote(row.timeout ? row.timeout.toISOString() : null)},
        ${sqlQuote(row.conversationId ? `sbx-ux-${row.id.slice(0, 8)}` : null)},
        NULL,
        ${sqlQuote(row.bashCommandId)},
        NULL,
        ${sqlQuote(row.cost)}
      );
    `);
  }

  return rows.length;
}

function decorateAutomation(automationId, seed) {
  const hexId = toHexUuid(automationId);
  const trigger = {
    ...seed.trigger,
    ...(seed.scheduleHuman ? { schedule_human: seed.scheduleHuman } : {}),
  };
  const lastTriggered =
    seed.lastTriggeredHoursAgo == null
      ? null
      : hoursAgo(seed.lastTriggeredHoursAgo).toISOString();

  sqlite(`
    UPDATE automations
    SET
      trigger = ${sqlQuote(JSON.stringify(trigger))},
      last_triggered_at = ${sqlQuote(lastTriggered)},
      updated_at = ${sqlQuote(new Date().toISOString())}
    WHERE id = ${sqlQuote(hexId)};
  `);
}

async function createSeed(seed) {
  const created = await api("/api/automation/v1/preset/prompt", {
    method: "POST",
    body: {
      name: seed.name,
      prompt: seed.prompt,
      trigger: seed.trigger,
      ...(seed.model ? { model: seed.model } : {}),
      ...(seed.timeout != null ? { timeout: seed.timeout } : {}),
      ...(seed.repos ? { repos: seed.repos } : {}),
    },
  });

  if (created.enabled !== seed.enabled) {
    await api(`/api/automation/v1/${created.id}`, {
      method: "PATCH",
      body: { enabled: seed.enabled },
    });
  }

  decorateAutomation(created.id, seed);
  const runCount = insertRuns(created.id, seed);
  console.log(
    `seeded ${seed.name} (${created.id}) — ${runCount} runs, enabled=${seed.enabled}`,
  );
  return created;
}

async function main() {
  const health = await api("/api/automation/health");
  if (health.status !== "ok") {
    throw new Error(`Automation backend is not healthy: ${JSON.stringify(health)}`);
  }

  await deleteExistingSeeds();
  for (const seed of SEEDS) {
    await createSeed(seed);
  }

  const list = await api("/api/automation/v1?limit=100");
  console.log(`\n${list.total} automations ready at ${BASE_URL}/automations`);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
