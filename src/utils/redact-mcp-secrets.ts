import type { MCPServerConfig } from "#/types/mcp-server";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";

/**
 * Minimum length for a config value to be treated as a redactable secret.
 * Shorter strings are too likely to collide with ordinary words in an
 * error message and would mangle it without hiding anything meaningful.
 */
const MIN_SECRET_LENGTH = 4;

const SECRETLIKE_PARAM_NAME = /token|key|secret|auth/i;

/**
 * Token shapes that may appear in backend error text even when the browser
 * never held the plaintext value (stored secrets round-trip encrypted and
 * are decrypted server-side, so a spawn/connect error can echo them back).
 */
const GENERIC_SECRET_PATTERNS: RegExp[] = [
  // GitHub personal access tokens (classic and fine-grained).
  /\bgh[pousr]_[A-Za-z0-9]{16,}\b/g,
  /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g,
  // Slack workspace tokens (xoxb-, xoxp-, xoxs-, ...).
  /\bxox[a-z](?:-[A-Za-z0-9]+)+/g,
  // Linear API keys.
  /\blin_api_[A-Za-z0-9]{10,}\b/g,
  // JWT-shaped values (three base64url segments starting with "eyJ").
  /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b/g,
];

// "Bearer <token>" keeps its prefix so the message stays readable.
const BEARER_PATTERN = /\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/gi;

function addValue(values: Set<string>, value: unknown): void {
  if (typeof value !== "string") return;
  if (value.length < MIN_SECRET_LENGTH) return;
  if (value === REDACTED_MCP_SECRET_VALUE) return;
  values.add(value);
}

function addRecordValues(
  values: Set<string>,
  record: Record<string, unknown> | null | undefined,
): void {
  for (const value of Object.values(record ?? {})) {
    addValue(values, value);
  }
}

function addUrlSecrets(values: Set<string>, rawUrl: string | undefined): void {
  if (!rawUrl) return;
  try {
    const url = new URL(rawUrl);
    addValue(values, url.username);
    addValue(values, url.password);
    addValue(values, decodeURIComponent(url.username));
    addValue(values, decodeURIComponent(url.password));
    url.searchParams.forEach((value, name) => {
      if (SECRETLIKE_PARAM_NAME.test(name)) addValue(values, value);
    });
  } catch {
    // Not a parseable URL — nothing to collect.
  }
}

function addAuthSecrets(
  values: Set<string>,
  auth: MCPServerConfig["auth"],
): void {
  if (!auth) return;
  switch (auth.strategy) {
    case "api_key":
    case "bearer":
      addValue(values, auth.value);
      break;
    case "basic":
      addValue(values, auth.username);
      addValue(values, auth.password);
      break;
    case "header":
      addRecordValues(values, auth.headers);
      break;
    case "oauth2":
      addValue(values, auth.authentication?.client_secret);
      addRecordValues(values, auth.state?.tokens);
      break;
    default:
      break;
  }
}

/**
 * Collect every secret VALUE a server config may carry, longest first so
 * replacing a shorter secret can never split a longer one that contains it.
 */
export function collectMcpSecretValues(server: MCPServerConfig): string[] {
  const values = new Set<string>();
  addRecordValues(values, server.env);
  addRecordValues(values, server.headers);
  addAuthSecrets(values, server.auth);
  addUrlSecrets(values, server.url);
  return [...values].sort((a, b) => b.length - a.length);
}

/**
 * Scrub secrets from MCP test/probe error text before it is displayed.
 * Replaces every known secret value from the given server config(s), then
 * applies generic token patterns as a safety net for plaintext the browser
 * never saw (e.g. values decrypted server-side and echoed in an error).
 */
export function redactMcpSecrets(
  text: string,
  ...servers: (MCPServerConfig | undefined)[]
): string {
  if (!text) return text;
  let redacted = text;
  for (const server of servers) {
    if (!server) continue;
    for (const value of collectMcpSecretValues(server)) {
      redacted = redacted.split(value).join(REDACTED_MCP_SECRET_VALUE);
    }
  }
  redacted = redacted.replace(
    BEARER_PATTERN,
    `Bearer ${REDACTED_MCP_SECRET_VALUE}`,
  );
  for (const pattern of GENERIC_SECRET_PATTERNS) {
    redacted = redacted.replace(pattern, REDACTED_MCP_SECRET_VALUE);
  }
  return redacted;
}
