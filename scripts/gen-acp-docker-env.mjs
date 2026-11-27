#!/usr/bin/env node
/**
 * Generate examples/acp-docker/.env from the single source of truth.
 *
 * Reads the version pins in config/defaults.json and writes the
 * `AGENT_SERVER_IMAGE=` line into examples/acp-docker/.env, pinning the
 * example to the exact `versions.agentServer` release. This keeps the
 * reproducible quickstart path on the SoT version instead of a hardcoded
 * tag that silently drifts below the Canvas compatibility floor
 * (compatibility.minimumAgentServer) and renders "Disconnected".
 *
 * The no-config `docker compose up` path uses the compose fallback
 * (`latest-python`, always >= the floor); running this script first pins the
 * example to the reproducible SoT version instead.
 *
 * Idempotent: re-running upserts the AGENT_SERVER_IMAGE line and leaves any
 * other lines in an existing .env untouched.
 *
 * Usage:
 *   node scripts/gen-acp-docker-env.mjs        # or: npm run example:acp-docker:env
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");

/**
 * @param {{ images: { agentServer: string }, versions: { agentServer: string } }} config
 * @returns {string} e.g. "ghcr.io/openhands/agent-server:1.28.1-python"
 */
export function computeAgentServerImage(config) {
  return `${config.images.agentServer}:${config.versions.agentServer}-python`;
}

/**
 * @param {{ images: { agentServer: string }, versions: { agentServer: string } }} config
 * @returns {string} the `AGENT_SERVER_IMAGE=<image>` line
 */
export function renderEnvLine(config) {
  return `AGENT_SERVER_IMAGE=${computeAgentServerImage(config)}`;
}

/**
 * Upsert the AGENT_SERVER_IMAGE line into an existing .env body, preserving
 * every other line. Appends the line if absent.
 * @param {string} existing prior .env contents ("" if the file is absent)
 * @param {string} line the `AGENT_SERVER_IMAGE=...` line to set
 * @returns {string} the updated .env contents
 */
export function upsertEnvLine(existing, line) {
  const eq = line.indexOf("=");
  if (eq <= 0) {
    // A keyless line ("", "novalue", "=value") would make `key` empty and
    // match every line — refuse rather than silently rewrite the whole file.
    throw new Error(
      `upsertEnvLine: expected a "KEY=value" line, got "${line}"`,
    );
  }
  const key = line.slice(0, eq + 1); // "AGENT_SERVER_IMAGE="
  const lines = existing.length ? existing.replace(/\n+$/, "").split("\n") : [];
  let replaced = false;
  const next = lines.map((l) => {
    if (l.startsWith(key)) {
      replaced = true;
      return line;
    }
    return l;
  });
  if (!replaced) next.push(line);
  return next.join("\n") + "\n";
}

function loadConfig() {
  return JSON.parse(
    readFileSync(join(projectRoot, "config", "defaults.json"), "utf-8"),
  );
}

function main() {
  const config = loadConfig();
  const line = renderEnvLine(config);
  const envPath = join(projectRoot, "examples", "acp-docker", ".env");

  let existing = "";
  try {
    existing = readFileSync(envPath, "utf-8");
  } catch {
    // no .env yet — create it
  }

  const updated = upsertEnvLine(existing, line);
  writeFileSync(envPath, updated);
  console.log(`Wrote ${line} to examples/acp-docker/.env`);
}

// Run main() only when invoked as a CLI. process.argv[1] is undefined in some
// ESM contexts (e.g. `node --input-type=module -e "import(...)"`), so guard it
// before pathToFileURL — otherwise importing this module for its exports throws
// ERR_INVALID_ARG_TYPE.
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main();
}
