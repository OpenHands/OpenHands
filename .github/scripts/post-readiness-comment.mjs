#!/usr/bin/env node

/**
 * Post (or update) a `ready-for-dev` readiness comment on an issue.
 *
 * Upserts by a hidden HTML marker so repeated runs update the same comment
 * instead of spamming duplicates. Uses the GITHUB_TOKEN available to the
 * workflow via the GH_TOKEN env var.
 *
 * Usage:
 *   node post-readiness-comment.mjs \
 *     --issue-number 123 \
 *     --repo OpenHands/OpenHands \
 *     --reasons-file /tmp/reasons.txt \
 *     --ready           # optional: post a "ready" message instead of "not ready"
 */

import { readFileSync } from "node:fs";

const MARKER = "<!-- issue-readiness-check -->";
const API_ROOT = process.env.GITHUB_API_URL ?? "https://api.github.com";

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const [rawKey, inlineValue] = arg.slice(2).split("=", 2);
    const key = rawKey.replaceAll("-", "_");
    if (inlineValue !== undefined) {
      args[key] = inlineValue;
      continue;
    }
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      args[key] = next;
      i += 1;
    } else {
      args[key] = "";
    }
  }
  return args;
}

function requireValue(name, value) {
  if (!value) throw new Error(`Missing required value: ${name}`);
  return value;
}

async function listComments(repo, issueNumber, token) {
  // Paginate through all comments so the marker is found even on issues
  // with >100 comments.
  const all = [];
  let page = 1;
  while (true) {
    const url = `${API_ROOT}/repos/${repo}/issues/${issueNumber}/comments?per_page=100&page=${page}`;
    const resp = await fetch(url, {
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
      },
    });
    if (!resp.ok) {
      throw new Error(`Failed to list comments: ${resp.status} ${await resp.text()}`);
    }
    const batch = await resp.json();
    all.push(...batch);
    if (batch.length < 100) break;
    page += 1;
  }
  return all;
}

async function createComment(repo, issueNumber, body, token) {
  const url = `${API_ROOT}/repos/${repo}/issues/${issueNumber}/comments`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ body }),
  });
  if (!resp.ok) {
    throw new Error(`Failed to create comment: ${resp.status} ${await resp.text()}`);
  }
  return resp.json();
}

async function updateComment(repo, commentId, body, token) {
  const url = `${API_ROOT}/repos/${repo}/issues/comments/${commentId}`;
  const resp = await fetch(url, {
    method: "PATCH",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ body }),
  });
  if (!resp.ok) {
    throw new Error(`Failed to update comment: ${resp.status} ${await resp.text()}`);
  }
  return resp.json();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const issueNumber = requireValue("issue-number", args.issue_number);
  const repo = requireValue("repo", args.repo);
  const token = process.env.GH_TOKEN;
  if (!token) throw new Error("GH_TOKEN env var is required");

  const isReady = args.ready === "" || args.ready === "true";
  const reasonsFile = args.reasons_file || args.reasons;
  const reasons = reasonsFile
    ? readFileSync(reasonsFile, "utf8")
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
    : [];

  const reasonList = reasons.length
    ? reasons.map((r) => `- ${r}`).join("\n")
    : "- The issue does not meet the ready-for-dev criteria.";

  const body = isReady
    ? [
        MARKER,
        "### ✅ `ready-for-dev` label applied",
        "",
        "This issue meets the readiness criteria and is ready for contribution.",
        "A contributor can now open a PR that links this issue (`Fixes #<number>`).",
        "",
        "_This is an automated check — no AI was used to generate this comment._",
      ].join("\n")
    : [
        MARKER,
        "### ⚠️ Not yet `ready-for-dev`",
        "",
        "This issue does not yet meet the readiness criteria, so the",
        "`ready-for-dev` label has not been applied (or has been removed).",
        "The criteria are type-specific:",
        "",
        "**Bug reports** (`bug` label):",
        "- The `### Actual Behavior` section must reference a supported run method",
        "  (`agent-canvas`, `npm run`, or `app.all-hands.dev/canvas`) **and** include",
        "  a screenshot or video of the bug.",
        "- An `### Acceptance Criteria` section with at least one checklist item (`- [ ] …`).",
        "",
        "**Enhancements** (`enhancement` label):",
        "- A `### Desired Behavior` section.",
        "- An `### Acceptance Criteria` section with at least one checklist item.",
        "",
        "What is missing:",
        "",
        reasonList,
        "",
        "Edit the issue to address the gaps and the `ready-for-dev` label will be",
        "applied automatically.",
        "",
        "_This is an automated check — no AI was used to generate this comment._",
      ].join("\n");

  const comments = await listComments(repo, issueNumber, token);
  const existing = comments.find((comment) => comment.body?.includes(MARKER));

  if (existing) {
    await updateComment(repo, existing.id, body, token);
    console.log(`Updated readiness comment ${existing.id} on issue #${issueNumber}`);
  } else {
    const created = await createComment(repo, issueNumber, body, token);
    console.log(`Created readiness comment ${created.id} on issue #${issueNumber}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
