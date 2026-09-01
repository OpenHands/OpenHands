#!/usr/bin/env node
/** Build one Docker target from the repository's pinned versions. */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(scriptDirectory, "..");
const config = JSON.parse(
  readFileSync(join(projectRoot, "config", "defaults.json"), "utf-8"),
);
const packageJson = JSON.parse(
  readFileSync(join(projectRoot, "package.json"), "utf-8"),
);

const supportedTargets = new Set([
  "final",
  "control-plane",
  "dev-sandbox",
  "enterprise-sandbox",
]);

function optionValue(args, index, option) {
  const value = args[index + 1];
  if (!value || value === "--") throw new Error(`${option} requires a value`);
  return value;
}

export function deriveAgentServerVersion(image, fallback) {
  const tag = image.match(/:([^/:@]+)$/)?.[1];
  return tag?.replace(/-python$/, "") || fallback;
}

export function parseArgs(args) {
  let target = "final";
  let tag;
  let platform;
  let agentServerImage = `${config.images.agentServer}:${config.versions.agentServer}-python`;
  const extraArgs = [];

  for (let index = 0; index < args.length; index++) {
    const argument = args[index];
    if (argument === "--") {
      extraArgs.push(...args.slice(index + 1));
      break;
    }
    if (argument === "--enterprise") {
      target = "enterprise-sandbox";
    } else if (argument === "--target") {
      target = optionValue(args, index, argument);
      index++;
    } else if (argument === "--tag") {
      tag = optionValue(args, index, argument);
      index++;
    } else if (argument === "--platform") {
      platform = optionValue(args, index, argument);
      index++;
    } else if (argument === "--agent-server-image") {
      agentServerImage = optionValue(args, index, argument);
      index++;
    } else {
      // Preserve the previous helper's passthrough behavior for Docker flags.
      extraArgs.push(argument);
    }
  }

  if (!supportedTargets.has(target)) {
    throw new Error(`Unsupported Docker target: ${target}`);
  }

  const agentServerVersion = deriveAgentServerVersion(
    agentServerImage,
    config.versions.agentServer,
  );
  if (target === "enterprise-sandbox") {
    platform ??= "linux/amd64";
    if (platform !== "linux/amd64") {
      throw new Error("The Enterprise sandbox image must target linux/amd64");
    }
    tag ??= `agent-canvas-enterprise-sandbox:${packageJson.version}-agent-server-${agentServerVersion}`;
  } else if (target === "control-plane") {
    tag ??= "agent-canvas-control-plane:local";
  } else if (target === "dev-sandbox") {
    tag ??= "agent-canvas-dev-sandbox:local";
  } else {
    tag ??= "agent-canvas:local";
  }

  return {
    agentServerImage,
    agentServerVersion,
    automationVersion: config.versions.automation,
    canvasBasePath: config.paths.canvasBasePath,
    canvasVersion: packageJson.version,
    extraArgs,
    platform,
    tag,
    target,
  };
}

export function buildDockerCommand(options) {
  const command = [
    "docker",
    "build",
    "-f",
    "docker/Dockerfile",
    "--target",
    options.target,
  ];
  if (options.platform) command.push("--platform", options.platform);
  command.push(
    "--build-arg",
    `AGENT_SERVER_IMAGE=${options.agentServerImage}`,
    "--build-arg",
    `AGENT_SERVER_VERSION=${options.agentServerVersion}`,
    "--build-arg",
    `AUTOMATION_VERSION=${options.automationVersion}`,
    "--build-arg",
    `AGENT_CANVAS_VERSION=${options.canvasVersion}`,
    "--build-arg",
    `VITE_BASE_PATH=${options.canvasBasePath}`,
    "-t",
    options.tag,
    ...options.extraArgs,
    ".",
  );
  return command;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const command = buildDockerCommand(options);
  console.log(`Docker target           : ${options.target}`);
  console.log(`Platform                : ${options.platform || "host default"}`);
  console.log(`Agent Server image      : ${options.agentServerImage}`);
  console.log(`Agent Server version    : ${options.agentServerVersion}`);
  console.log(`Automation version      : ${options.automationVersion}`);
  console.log(`Canvas base path        : ${options.canvasBasePath}`);
  console.log(`Tag                     : ${options.tag}`);
  console.log(`\n$ ${command.join(" ")}\n`);
  try {
    execFileSync(command[0], command.slice(1), {
      cwd: projectRoot,
      stdio: "inherit",
    });
  } catch (error) {
    process.exit(error.status || 1);
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main();
}
