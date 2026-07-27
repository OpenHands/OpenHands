#!/usr/bin/env node
/**
 * Local Docker build helper.
 *
 * Reads version pins from config/defaults.json and invokes `docker build`
 * with the correct --build-arg values so developers never need to remember
 * (or hardcode) version strings.
 *
 * Usage:
 *   node scripts/docker-build.mjs                      # defaults
 *   node scripts/docker-build.mjs --tag my-tag          # custom tag
 *   node scripts/docker-build.mjs -- --no-cache         # extra docker args
 */
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");

const config = JSON.parse(
  readFileSync(join(projectRoot, "config", "defaults.json"), "utf-8"),
);

const agentServerImage = `${config.images.agentServer}:${config.versions.agentServer}-python`;
const automationVersion = config.versions.automation;
const canvasBasePath = config.paths.canvasBasePath;

// Parse CLI: --tag <name> and everything after -- is passed to docker build
let tag = "agent-canvas:local";
const extraArgs = [];
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--tag" && i + 1 < args.length) {
    tag = args[++i];
  } else if (args[i] === "--") {
    extraArgs.push(...args.slice(i + 1));
    break;
  } else {
    extraArgs.push(args[i]);
  }
}

const cmd = [
  "docker",
  "build",
  "-f",
  "docker/Dockerfile",
  "--build-arg",
  `AGENT_SERVER_IMAGE=${agentServerImage}`,
  "--build-arg",
  `AUTOMATION_VERSION=${automationVersion}`,
  "--build-arg",
  `VITE_BASE_PATH=${canvasBasePath}`,
  "-t",
  tag,
  ...extraArgs,
  ".",
];

console.log(`Agent Server image      : ${agentServerImage}`);
console.log(`Automation version      : ${automationVersion}`);
console.log(`Canvas base path        : ${canvasBasePath}`);
console.log(`Tag                     : ${tag}`);
console.log(`\n$ ${cmd.join(" ")}\n`);

try {
  execFileSync(cmd[0], cmd.slice(1), {
    cwd: projectRoot,
    stdio: "inherit",
  });
} catch (err) {
  process.exit(err.status || 1);
}
