#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";

const REQUIRED_LLM_API_KEY_ENV_VARS = [
  "LIVE_E2E_LLM_API_KEY",
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "LLM_API_KEY",
];
const DEFAULT_PROXY_BASE_URL = "https://llm-proxy.app.all-hands.dev";
const DEFAULT_PROXY_MODEL = "openhands/claude-haiku-4-5-20251001";
const DEFAULT_OPENAI_MODEL = "openai/gpt-5.4-mini";
const DEFAULT_ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5-20251001";
const DEFAULT_BACKEND_URL = "http://127.0.0.1:18100";
const DEFAULT_FRONTEND_PORT = "3101";
const PLAYWRIGHT_CONFIG = "playwright.live.config.ts";

let generatedSessionApiKey = false;

function hasValue(name) {
  return Boolean(process.env[name]?.trim());
}

function firstConfiguredEnvVar(names) {
  return names.find((name) => hasValue(name)) ?? "";
}

function commandExists(command) {
  const result = spawnSync(platformCommand(command), ["--version"], {
    stdio: "ignore",
  });
  return !result.error && result.status === 0;
}

function platformCommand(command) {
  return process.platform === "win32" ? `${command}.cmd` : command;
}

function localPlaywrightExists() {
  const executable =
    process.platform === "win32" ? "playwright.cmd" : "playwright";
  return existsSync(
    path.join(process.cwd(), "node_modules", ".bin", executable),
  );
}

function usesProxyDefaults(apiKeySource) {
  return (
    apiKeySource === "LIVE_E2E_LLM_API_KEY" ||
    (!hasValue("OPENAI_API_KEY") &&
      !hasValue("ANTHROPIC_API_KEY") &&
      apiKeySource === "LLM_API_KEY")
  );
}

function resolvedLLMBaseUrl(apiKeySource) {
  if (hasValue("LIVE_E2E_LLM_BASE_URL")) {
    return process.env.LIVE_E2E_LLM_BASE_URL.trim();
  }
  return usesProxyDefaults(apiKeySource) ? DEFAULT_PROXY_BASE_URL : "(unset)";
}

function resolvedLLMModel(apiKeySource) {
  if (!apiKeySource) {
    return "(depends on credential source)";
  }
  if (hasValue("LIVE_E2E_LLM_MODEL")) {
    return process.env.LIVE_E2E_LLM_MODEL.trim();
  }
  if (resolvedLLMBaseUrl(apiKeySource) !== "(unset)") {
    return DEFAULT_PROXY_MODEL;
  }
  if (apiKeySource === "OPENAI_API_KEY") {
    return DEFAULT_OPENAI_MODEL;
  }
  return DEFAULT_ANTHROPIC_MODEL;
}

function redactUrlForLog(value) {
  if (!value || value.startsWith("(")) {
    return value;
  }

  try {
    const url = new URL(value);
    if (url.username || url.password) {
      url.username = "redacted";
      url.password = "redacted";
    }
    return url.toString();
  } catch {
    return value.replace(/\/\/[^/@\s]+@/g, "//redacted:redacted@");
  }
}

function ensureSessionApiKey() {
  if (hasValue("LIVE_E2E_SESSION_API_KEY")) {
    return;
  }
  process.env.LIVE_E2E_SESSION_API_KEY = randomBytes(32).toString("hex");
  generatedSessionApiKey = true;
}

function printUsage() {
  console.log(`
Run the live Agent Server end-to-end test locally.

Usage:
  npm run test:e2e:live
  npm run test:e2e:live -- --headed
  npm run test:e2e:live -- --debug
  npm run test:e2e:live -- --check

Required:
  Set one LLM credential before running:
  - LIVE_E2E_LLM_API_KEY
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - LLM_API_KEY

Optional:
  - LIVE_E2E_LLM_BASE_URL
  - LIVE_E2E_LLM_MODEL
  - LIVE_E2E_SESSION_API_KEY (generated per run when unset)
  - LIVE_E2E_BACKEND_URL
  - LIVE_E2E_FRONTEND_PORT

The npm script loads .env automatically through Node's --env-file-if-exists flag.
`);
}

function printConfiguration(apiKeySource) {
  const llmBaseUrl = resolvedLLMBaseUrl(apiKeySource);
  console.log("Live Agent Server E2E configuration:");
  console.log(`- LLM API key source: ${apiKeySource || "(missing)"}`);
  console.log(`- LIVE_E2E_LLM_BASE_URL: ${redactUrlForLog(llmBaseUrl)}`);
  console.log(`- LIVE_E2E_LLM_MODEL: ${resolvedLLMModel(apiKeySource)}`);
  console.log(
    `- LIVE_E2E_SESSION_API_KEY: ${
      hasValue("LIVE_E2E_SESSION_API_KEY")
        ? generatedSessionApiKey
          ? "(generated for this run)"
          : "(configured)"
        : "(missing)"
    }`,
  );
  console.log(
    `- LIVE_E2E_BACKEND_URL: ${
      hasValue("LIVE_E2E_BACKEND_URL")
        ? redactUrlForLog(process.env.LIVE_E2E_BACKEND_URL.trim())
        : `(default: ${DEFAULT_BACKEND_URL})`
    }`,
  );
  console.log(
    `- LIVE_E2E_FRONTEND_PORT: ${
      hasValue("LIVE_E2E_FRONTEND_PORT")
        ? redactUrlForLog(process.env.LIVE_E2E_FRONTEND_PORT.trim())
        : `(default: ${DEFAULT_FRONTEND_PORT})`
    }`,
  );
}

function validateEnvironment() {
  ensureSessionApiKey();
  const apiKeySource = firstConfiguredEnvVar(REQUIRED_LLM_API_KEY_ENV_VARS);
  const errors = [];

  if (!apiKeySource) {
    errors.push(
      [
        "Missing LLM credential.",
        `Set one of: ${REQUIRED_LLM_API_KEY_ENV_VARS.join(", ")}.`,
        "For the hosted LLM proxy, use LIVE_E2E_LLM_API_KEY or LLM_API_KEY.",
      ].join(" "),
    );
  }

  if (!localPlaywrightExists()) {
    errors.push(
      "Missing local Playwright install. Run `npm ci` before running live E2E.",
    );
  }

  if (!commandExists("uvx")) {
    errors.push(
      [
        "Missing `uvx`, which `npm run dev:minimal` uses to start the real Agent Server.",
        "Install uv with: `curl -LsSf https://astral.sh/uv/install.sh | sh`.",
      ].join(" "),
    );
  }

  printConfiguration(apiKeySource);

  if (errors.length === 0) {
    console.log("Environment check passed.");
    return true;
  }

  console.error("");
  console.error("Live Agent Server E2E is not ready to run:");
  for (const error of errors) {
    console.error(`- ${error}`);
  }
  console.error("");
  console.error("After fixing the above, run `npm run test:e2e:live` again.");
  return false;
}

async function runPlaywright(args) {
  const child = spawn(
    platformCommand("npx"),
    ["playwright", "test", ...args, `--config=${PLAYWRIGHT_CONFIG}`],
    {
      stdio: "inherit",
    },
  );

  const exitCode = await new Promise((resolve) => {
    child.on("exit", (code) => resolve(code ?? 1));
    child.on("error", () => resolve(1));
  });

  process.exit(exitCode);
}

const args = process.argv.slice(2);

if (args.includes("--help") || args.includes("-h")) {
  printUsage();
  process.exit(0);
}

const checkOnly = args.includes("--check");
const playwrightArgs = args.filter((arg) => arg !== "--check");
const isValid = validateEnvironment();

if (!isValid) {
  process.exit(1);
}

if (checkOnly) {
  process.exit(0);
}

await runPlaywright(playwrightArgs);
