#!/usr/bin/env node

import { existsSync, readFileSync, realpathSync } from "node:fs";
import { isAbsolute, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const DEFAULT_MARKER = "<!-- agent-canvas-mock-llm-e2e-report -->";
const API_ROOT = process.env.GITHUB_API_URL ?? "https://api.github.com";

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

function requireValue(name, value) {
  if (!value) {
    throw new Error(`Missing required value: ${name}`);
  }
  return value;
}

function assertWithinCwd(label, path) {
  const relativePath = relative(realpathSync(resolve(".")), path);
  if (relativePath.startsWith("..") || isAbsolute(relativePath)) {
    throw new Error(`${label} must be within the current working directory.`);
  }
}

function resolveWithinCwd(label, filePath) {
  const resolvedPath = resolve(filePath);
  if (!existsSync(resolvedPath)) {
    throw new Error(`${label} does not exist: ${resolvedPath}`);
  }

  assertWithinCwd(label, realpathSync(resolvedPath));
  return resolvedPath;
}

function validateRepo(repo) {
  if (!/^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\/[a-zA-Z0-9._-]+$/.test(repo)) {
    throw new Error(`Invalid repo format: ${repo}`);
  }
  return repo;
}

function validateIssueNumber(issueNumber) {
  if (!/^\d+$/.test(String(issueNumber))) {
    throw new Error(`Invalid issue number: ${issueNumber}`);
  }
  return String(issueNumber);
}

function sleep(ms) {
  return new Promise((resolvePromise) => {
    setTimeout(resolvePromise, ms);
  });
}

function retryDelayMs(attempt, response) {
  const retryAfter = response.headers.get("retry-after");
  if (retryAfter && /^\d+$/.test(retryAfter)) {
    return Number(retryAfter) * 1000;
  }
  return 1000 * 2 ** attempt;
}

function parseGitHubPayload(method, path, status, text) {
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(
      `GitHub API ${method} ${path} returned invalid JSON with status ${status}: ${text.slice(
        0,
        500,
      )}`,
    );
  }
}

async function githubRequest(method, path, token, body) {
  const retryableStatuses = new Set([429, 502, 503, 504]);

  for (let attempt = 0; attempt < 5; attempt += 1) {
    const response = await fetch(`${API_ROOT}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });

    const text = await response.text();
    if (!response.ok && retryableStatuses.has(response.status) && attempt < 4) {
      await sleep(retryDelayMs(attempt, response));
      continue;
    }

    const payload = parseGitHubPayload(method, path, response.status, text);
    if (!response.ok) {
      throw new Error(
        `GitHub API ${method} ${path} failed with ${response.status}: ${text}`,
      );
    }
    return payload;
  }

  throw new Error(`GitHub API ${method} ${path} failed after retries.`);
}

async function listIssueComments(repo, issueNumber, token) {
  const comments = [];
  for (let page = 1; page <= 10; page += 1) {
    const batch = await githubRequest(
      "GET",
      `/repos/${repo}/issues/${issueNumber}/comments?per_page=100&page=${page}`,
      token,
    );
    comments.push(...batch);
    if (batch.length < 100) {
      break;
    }
  }
  return comments;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function buildCommentBody(body, marker = DEFAULT_MARKER) {
  if (body.includes(marker)) {
    return body;
  }
  return `${marker}\n${body}`;
}

export function isMatchingJobComment(comment, { marker, legacyTitle }) {
  const body = comment.body ?? "";
  if (body.includes(marker)) {
    return true;
  }

  if (!legacyTitle) {
    return false;
  }

  const user = comment.user ?? {};
  const isBot = user.type === "Bot" || user.login?.endsWith("[bot]");
  if (!isBot) {
    return false;
  }

  const legacyHeading = new RegExp(
    `^##\\s+\\S+\\s+${escapeRegExp(legacyTitle)}\\s*$`,
    "m",
  );
  return legacyHeading.test(body);
}

export function findMatchingJobComments(comments, options) {
  return comments.filter((comment) => isMatchingJobComment(comment, options));
}

export async function replaceJobComment({
  repo,
  issueNumber,
  token,
  body,
  marker = DEFAULT_MARKER,
  legacyTitle = "",
}) {
  const comments = await listIssueComments(repo, issueNumber, token);
  const existing = findMatchingJobComments(comments, { marker, legacyTitle });

  for (const comment of existing) {
    await githubRequest(
      "DELETE",
      `/repos/${repo}/issues/comments/${comment.id}`,
      token,
    );
  }

  const created = await githubRequest(
    "POST",
    `/repos/${repo}/issues/${issueNumber}/comments`,
    token,
    { body: buildCommentBody(body, marker) },
  );

  return { deleted: existing.length, created };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const issueNumber =
    args.issue_number ??
    process.env.PR_NUMBER ??
    process.env.ISSUE_NUMBER ??
    "";

  if (!issueNumber) {
    console.log("Skipping PR comment because no PR number was provided.");
    return;
  }

  const repo = validateRepo(
    requireValue("repo", args.repo ?? process.env.GITHUB_REPOSITORY),
  );
  const validatedIssueNumber = validateIssueNumber(issueNumber);
  const token = requireValue(
    "token",
    args.token ?? process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN,
  );
  const marker = args.marker ?? DEFAULT_MARKER;
  const bodyFile = resolveWithinCwd(
    "body-file",
    requireValue("body-file", args.body_file),
  );
  const body = readFileSync(bodyFile, "utf8");

  const result = await replaceJobComment({
    repo,
    issueNumber: validatedIssueNumber,
    token,
    body,
    marker,
    legacyTitle: args.legacy_title ?? "",
  });

  console.log(
    `Deleted ${result.deleted} existing PR comment${
      result.deleted === 1 ? "" : "s"
    }; created PR comment ${result.created.id}.`,
  );
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
