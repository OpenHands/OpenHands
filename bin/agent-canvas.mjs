#!/usr/bin/env node
/**
 * CLI entry point for @openhands/agent-canvas
 *
 * Runs the full Agent Canvas stack locally by default:
 * - Agent-server via uvx
 * - Automation backend via uvx
 * - Pre-built static frontend
 *
 * This is the production equivalent of `npm run dev` - it runs the full stack
 * but serves pre-built static assets instead of the Vite dev server.
 */

import { existsSync, readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PKG_JSON = join(__dirname, "..", "package.json");
// Build output is in build/ (not build/client/) - see react-router.config.ts unpackClientDirectory
const BUILD_DIR = join(__dirname, "..", "build");

const DEFAULTS_JSON = join(__dirname, "..", "config", "defaults.json");

// Check for version/help/info/public flags first
const args = process.argv.slice(2);
if (args.includes("-v") || args.includes("--version")) {
  const { version } = JSON.parse(readFileSync(PKG_JSON, "utf-8"));
  console.log(version);
  process.exit(0);
}
if (args.includes("--info")) {
  const { version } = JSON.parse(readFileSync(PKG_JSON, "utf-8"));
  const defaults = JSON.parse(readFileSync(DEFAULTS_JSON, "utf-8"));
  console.log(`@openhands/agent-canvas ${version}

Default stack versions:
  agent-server:    ${defaults.versions.agentServer}
  automation:      ${defaults.versions.automation}

Compatibility:
  agent-server:    >= ${defaults.compatibility.minimumAgentServer}

Default ports:
  ingress:         ${defaults.ports.proxy}
  agent-server:    ${defaults.ports.agentServer}
  automation:      ${defaults.ports.automation}

Override versions via environment variables:
  OH_AGENT_SERVER_VERSION, OH_AGENT_SERVER_GIT_REF, OH_AGENT_SERVER_LOCAL_PATH
  OH_AUTOMATION_VERSION, OH_AUTOMATION_GIT_REF`);
  process.exit(0);
}
const isPublic = args.includes("--public");
const isFrontendOnly = args.includes("--frontend-only");
const isBackendOnly = args.includes("--backend-only");

if (args.includes("-h") || args.includes("--help")) {
  console.log(`
@openhands/agent-canvas - Run the Agent Canvas UI with agent-server

Runs the full stack with agent-server and automation backend via uvx,
and serves pre-built static frontend assets.

USAGE:
  npx @openhands/agent-canvas [options]

AUTH MODES:
  By default the server auto-generates an API key that is injected into
  the frontend at startup — no login required.

  --public    Enable public mode. Requires LOCAL_BACKEND_API_KEY env var.
              The key protects the server but is NOT injected into the
              frontend. Users must paste it when the UI loads.

OPTIONS:
  -p, --port <port>     Ingress port (default: 8000)
  --public              Enable public mode (see above)
  --frontend-only       Start only the static frontend behind ingress
  --backend-only        Start only agent-server + automation behind ingress
  -v, --version         Show version number
  --info                Show version and default stack configuration
  -h, --help            Show this help message

ENVIRONMENT VARIABLES:
  LOCAL_BACKEND_API_KEY        API key for the server. Required in --public
                               mode; optional otherwise (auto-generated if
                               omitted, persisted across restarts).
  OH_SECRET_KEY                Secret key for encrypting settings
  OH_AGENT_SERVER_GIT_REF      Git ref for agent-server
  OH_AGENT_SERVER_LOCAL_PATH   Path to local SDK checkout (for development)
  OH_AGENT_SERVER_VERSION      Specific PyPI version for agent-server

Note: LLM settings are configured through the web UI settings page,
not environment variables.

EXAMPLES:
  # Start full stack (local mode, auto-generated key)
  npx @openhands/agent-canvas

  # Pin a specific key (local mode, key auto-injected into frontend)
  LOCAL_BACKEND_API_KEY=my-key npx @openhands/agent-canvas

  # Public mode — users must enter the API key in the browser
  LOCAL_BACKEND_API_KEY=my-secret npx @openhands/agent-canvas --public

  # Use a specific port
  npx @openhands/agent-canvas --port 3000

  # Start only the static frontend behind ingress
  npx @openhands/agent-canvas --frontend-only

  # Start only the agent-server and automation backend behind ingress
  npx @openhands/agent-canvas --backend-only

  # Show default stack versions and ports
  npx @openhands/agent-canvas --info

  # Use local SDK checkout for development
  OH_AGENT_SERVER_LOCAL_PATH=/path/to/sdk npx @openhands/agent-canvas
`);
  process.exit(0);
}

if (isFrontendOnly && isBackendOnly) {
  console.error(
    "Error: --frontend-only and --backend-only cannot be used together",
  );
  process.exit(1);
}

if (isFrontendOnly && isPublic) {
  console.error("Error: --public cannot be used with --frontend-only");
  process.exit(1);
}

// Check build exists before doing anything else unless no frontend will run.
if (!isBackendOnly && !existsSync(BUILD_DIR)) {
  console.error(`
Error: No build found at ${BUILD_DIR}

This package needs to be built first. If you installed from npm,
this is a packaging error. If running from source:

  npm install
  npm run build
`);
  process.exit(1);
}

let main;
try {
  ({ main } = await import("../scripts/dev-with-automation.mjs"));
} catch (err) {
  console.error("Failed to load required scripts. Try reinstalling:");
  console.error("  npm install -g @openhands/agent-canvas@latest");
  console.error(`\nError: ${err.message}`);
  process.exit(1);
}

main({
  bannerTitle: "Agent Canvas",
  staticMode: true,
  staticDir: BUILD_DIR,
  mode: "agent-canvas",
  isPublic,
}).catch((err) => {
  console.error(`Fatal error: ${err.message}`);
  if (err.stack) {
    console.error(err.stack);
  }
  process.exit(1);
});
