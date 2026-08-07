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
import { existsSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");
const defaultTsClientPath = resolve(projectRoot, "..", "typescript-client");
const typescriptClientPath =
  process.env.TYPESCRIPT_CLIENT_PATH || defaultTsClientPath;
const defaultSdkPath = resolve(projectRoot, "..", "software-agent-sdk");
const softwareAgentSdkPath =
  process.env.SOFTWARE_AGENT_SDK_PATH ||
  process.env.OH_AGENT_SERVER_LOCAL_PATH ||
  defaultSdkPath;

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
];

// Local forks pin `@openhands/typescript-client` via file:../typescript-client.
// Docker needs that sibling as an extra build context so npm ci can resolve it.
if (existsSync(join(typescriptClientPath, "package.json"))) {
  cmd.push("--build-context", `typescript-client=${typescriptClientPath}`);
} else {
  console.warn(
    `WARNING: ${typescriptClientPath} not found.\n` +
      `  If package.json uses file:../typescript-client, the frontend build will fail.\n` +
      `  Set TYPESCRIPT_CLIENT_PATH or place the repo next to OpenHands.`,
  );
}

// Local software-agent-sdk with unreleased agent-server APIs (e.g. workspace clone).
if (
  existsSync(
    join(softwareAgentSdkPath, "openhands-agent-server", "pyproject.toml"),
  )
) {
  cmd.push("--build-context", `software-agent-sdk=${softwareAgentSdkPath}`);
} else {
  console.warn(
    `WARNING: ${softwareAgentSdkPath} not found.\n` +
      `  Without it, POST /api/workspaces/clone is missing unless AGENT_SERVER_SDK_GIT_REF is set.\n` +
      `  Set SOFTWARE_AGENT_SDK_PATH / OH_AGENT_SERVER_LOCAL_PATH or place the SDK next to OpenHands.`,
  );
}

cmd.push("-t", tag, ...extraArgs, ".");

console.log(`Agent Server image      : ${agentServerImage}`);
console.log(`Automation version      : ${automationVersion}`);
console.log(`Canvas base path        : ${canvasBasePath}`);
console.log(`TypeScript client       : ${typescriptClientPath}`);
console.log(`Software agent SDK      : ${softwareAgentSdkPath}`);
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
