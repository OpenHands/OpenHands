#!/usr/bin/env node
/** Start the separated local control-plane and Agent Server containers. */
import { execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { chmodSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
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

function newSecret() {
  return randomBytes(32).toString("hex");
}

function persistedSecretsPath(environment) {
  const stateDirectory =
    environment.OH_CANVAS_SAFE_STATE_DIR ||
    join(projectRoot, ".tmp", "docker-separated-state");
  return join(stateDirectory, "runtime-secrets.json");
}

function loadOrCreateSecrets(environment) {
  const secretsPath = persistedSecretsPath(environment);
  let persisted = {};
  try {
    persisted = JSON.parse(readFileSync(secretsPath, "utf-8"));
  } catch {
    // First start, or an unreadable file: create fresh local-only values.
  }

  const sessionKey =
    environment.LOCAL_BACKEND_API_KEY ||
    environment.OH_SESSION_API_KEYS_0 ||
    persisted.LOCAL_BACKEND_API_KEY ||
    newSecret();
  const secretKey =
    environment.OH_SECRET_KEY || persisted.OH_SECRET_KEY || newSecret();

  const valuesToPersist = { ...persisted };
  if (
    !environment.LOCAL_BACKEND_API_KEY &&
    !environment.OH_SESSION_API_KEYS_0
  ) {
    valuesToPersist.LOCAL_BACKEND_API_KEY = sessionKey;
  }
  if (!environment.OH_SECRET_KEY) {
    valuesToPersist.OH_SECRET_KEY = secretKey;
  }
  if (Object.keys(valuesToPersist).length > 0) {
    mkdirSync(dirname(secretsPath), { recursive: true });
    writeFileSync(
      secretsPath,
      `${JSON.stringify(valuesToPersist, null, 2)}\n`,
      { mode: 0o600 },
    );
    chmodSync(secretsPath, 0o600);
  }

  return { secretKey, sessionKey };
}

export function buildComposeCommand(args, baseEnvironment) {
  const sessionKey =
    baseEnvironment.LOCAL_BACKEND_API_KEY ||
    baseEnvironment.OH_SESSION_API_KEYS_0 ||
    newSecret();
  const secretKey = baseEnvironment.OH_SECRET_KEY || newSecret();
  const agentServerImage = `${config.images.agentServer}:${config.versions.agentServer}-python`;

  return {
    command: [
      "docker",
      "compose",
      "-f",
      "docker/compose.yml",
      "up",
      "--build",
      ...args,
    ],
    env: {
      ...baseEnvironment,
      AGENT_CANVAS_VERSION: packageJson.version,
      AGENT_SERVER_IMAGE: agentServerImage,
      AGENT_SERVER_VERSION: config.versions.agentServer,
      AUTOMATION_VERSION: config.versions.automation,
      LOCAL_BACKEND_API_KEY: sessionKey,
      OH_SECRET_KEY: secretKey,
      OH_SESSION_API_KEYS_0: sessionKey,
      VITE_BASE_PATH: config.paths.canvasBasePath,
    },
  };
}

function main() {
  const persisted = loadOrCreateSecrets(process.env);
  const invocation = buildComposeCommand(process.argv.slice(2), {
    ...process.env,
    LOCAL_BACKEND_API_KEY:
      process.env.LOCAL_BACKEND_API_KEY || persisted.sessionKey,
    OH_SECRET_KEY: process.env.OH_SECRET_KEY || persisted.secretKey,
  });
  console.log(
    `Starting separated Agent Canvas at http://localhost:${invocation.env.AGENT_CANVAS_PORT || config.ports.proxy}`,
  );
  execFileSync(invocation.command[0], invocation.command.slice(1), {
    cwd: projectRoot,
    env: invocation.env,
    stdio: "inherit",
  });
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main();
}
