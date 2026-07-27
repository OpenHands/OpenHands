/**
 * Playwright config for mock-LLM E2E tests against a Docker container.
 *
 * Reuses the same test specs as playwright.mock-llm.config.ts but launches
 * the agent-canvas stack inside a Docker container instead of via
 * bin/agent-canvas.mjs + uvx.
 *
 * Starts two processes:
 *   1. Mock LLM server (Python on the host, using openhands-sdk TestLLM)
 *   2. Docker container running the agent-canvas all-in-one image
 *      (agent-server + automation backend + static frontend + proxy)
 *      The container also starts a second static-server instance on
 *      PUBLIC_MODE_PORT with --auth-required for auth-mode E2E tests.
 *
 * Networking:
 *   Uses --network host on Linux so the container shares the host's network
 *   stack. This means the agent-server inside Docker can reach the mock LLM
 *   server at 127.0.0.1:<port> — identical to the npm path.
 *
 *   For macOS/Windows (Docker Desktop with bridge networking), set
 *   MOCK_LLM_AGENT_URL=http://host.docker.internal:<port> so the
 *   agent-server can reach the host-side mock LLM server.
 *
 * Required:
 *   - A built Docker image. Set MOCK_LLM_DOCKER_IMAGE to the image tag
 *     (default: ghcr.io/openhands/agent-canvas:latest).
 *   - Docker daemon must be running.
 */

import { defineConfig, devices } from "@playwright/test";
import { randomBytes } from "node:crypto";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

// ── Docker image ────────────────────────────────────────────────────────
const DOCKER_IMAGE =
  process.env.MOCK_LLM_DOCKER_IMAGE ?? "ghcr.io/openhands/agent-canvas:latest";

// Container name for cleanup — unique per run to avoid collisions.
const CONTAINER_NAME =
  process.env.MOCK_LLM_CONTAINER_NAME ??
  `agent-canvas-mock-llm-${randomBytes(4).toString("hex")}`;

// ── Port allocation (separate from live E2E / dev to avoid collisions) ─
const MOCK_LLM_PORT = process.env.MOCK_LLM_PORT ?? "9999";

// The Docker container exposes a single port for the unified ingress.
// With --network host this is accessible at localhost directly.
const INGRESS_PORT = process.env.MOCK_LLM_INGRESS_PORT ?? "18300";

// Public-mode static server — runs inside the Docker container when
// PUBLIC_MODE_PORT is set (see docker/entrypoint.sh). With --network host
// the port is accessible from the host at localhost directly.
const PUBLIC_MODE_PORT = process.env.MOCK_LLM_PUBLIC_MODE_PORT ?? "18301";

// ── Session API key ────────────────────────────────────────────────────
const sessionApiKey =
  process.env.MOCK_LLM_SESSION_API_KEY?.trim() ||
  randomBytes(32).toString("hex");
process.env.MOCK_LLM_SESSION_API_KEY = sessionApiKey;

// ── URLs ───────────────────────────────────────────────────────────────
// The Docker entrypoint starts static-server.mjs with `--host ::` (dual-
// stack), so it accepts both IPv4 and IPv6 connections — same as the npm
// path's ingress.mjs.  `localhost` is safe for both configs now.
// The mock LLM server (Python, host-side) also listens dual-stack.
const INGRESS_URL = `http://localhost:${INGRESS_PORT}/`;
const MOCK_LLM_URL = `http://127.0.0.1:${MOCK_LLM_PORT}`;

// Python binary for the mock server — defaults to "python3" but CI can
// point this at a venv (e.g. ".mock-llm-venv/bin/python3") to avoid
// PEP 668 "externally managed" errors on Ubuntu 24.04+.
const MOCK_LLM_PYTHON = process.env.MOCK_LLM_PYTHON ?? "python3";

// Export for the test helpers — BACKEND_URL points to the ingress (API
// calls are proxied to the agent-server, so no direct backend port needed).
process.env.MOCK_LLM_BACKEND_URL = `http://localhost:${INGRESS_PORT}`;
process.env.MOCK_LLM_PORT = MOCK_LLM_PORT;
process.env.MOCK_LLM_PUBLIC_MODE_URL = `http://localhost:${PUBLIC_MODE_PORT}`;
process.env.VITE_SESSION_API_KEY = sessionApiKey;

// MOCK_LLM_AGENT_URL — the URL the agent-server inside Docker uses to
// call the mock LLM for inference. With --network host on Linux the
// agent-server can reach 127.0.0.1 directly. For macOS/Windows Docker
// Desktop, override this to http://host.docker.internal:<port>.
if (!process.env.MOCK_LLM_AGENT_URL) {
  process.env.MOCK_LLM_AGENT_URL = MOCK_LLM_URL;
}

// ── Skill test support ────────────────────────────────────────────────
// The skill tests create git repos and user skill files on the host.
// We volume-mount these into the container so the agent-server can see them.

// Project skills: host creates repos here, container sees them at a fixed path.
const SKILL_REPOS_HOST_DIR = resolve(".tmp/mock-llm-skill-repos");
const SKILL_REPOS_CONTAINER_DIR = "/tmp/mock-llm-skill-repos";
mkdirSync(SKILL_REPOS_HOST_DIR, { recursive: true });
process.env.MOCK_LLM_SKILL_REPOS_CONTAINER_DIR = SKILL_REPOS_CONTAINER_DIR;

// User skills: host creates skill files here, container mounts them at
// the agent-server's expected ~/.openhands/skills/ path.
const USER_SKILLS_HOST_DIR = resolve(".tmp/mock-llm-user-skills");
const USER_SKILLS_CONTAINER_DIR = "/home/openhands/.openhands/skills";
mkdirSync(USER_SKILLS_HOST_DIR, { recursive: true });
process.env.MOCK_LLM_USER_SKILLS_HOST_DIR = USER_SKILLS_HOST_DIR;

// ── Folder-workspace test support ──────────────────────────────────────
// The folder-workspace test creates a temp directory on the host that the
// agent-server's folder browser needs to list. Mount a shared directory into
// the container at the same path so the agent-server can see it.
const FOLDER_WORKSPACE_HOST_DIR = resolve(".tmp/e2e-folder-workspace-test");
const FOLDER_WORKSPACE_CONTAINER_DIR = "/tmp/e2e-folder-workspace-test";
mkdirSync(FOLDER_WORKSPACE_HOST_DIR, { recursive: true });
process.env.MOCK_LLM_FOLDER_WORKSPACE_HOST_DIR = FOLDER_WORKSPACE_HOST_DIR;
process.env.MOCK_LLM_FOLDER_WORKSPACE_CONTAINER_DIR =
  FOLDER_WORKSPACE_CONTAINER_DIR;

// ── ACP test support ──────────────────────────────────────────────────
// The mock ACP server script lives on the host. We volume-mount it into
// the container and tell the test which container-side paths to use when
// typing the `acp_command` into the Settings UI.
const MOCK_ACP_HOST_PATH = resolve(
  "tests/e2e/mock-llm/scripts/mock-acp-server.py",
);
const MOCK_ACP_CONTAINER_PATH = "/opt/mock-acp-server.py";

// The agent-server image ships Python 3 via the openhands-sdk base.
process.env.MOCK_ACP_CONTAINER_PYTHON = "python3";
process.env.MOCK_ACP_CONTAINER_SCRIPT = MOCK_ACP_CONTAINER_PATH;

const DEFAULT_CI_GLOBAL_TIMEOUT_MS = 1_200_000;
const configuredCiGlobalTimeoutMs = Number.parseInt(
  process.env.MOCK_LLM_DOCKER_GLOBAL_TIMEOUT_MS ??
    String(DEFAULT_CI_GLOBAL_TIMEOUT_MS),
  10,
);
const ciGlobalTimeoutMs = Number.isFinite(configuredCiGlobalTimeoutMs)
  ? configuredCiGlobalTimeoutMs
  : DEFAULT_CI_GLOBAL_TIMEOUT_MS;

export default defineConfig({
  testDir: "./tests/e2e/mock-llm",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  // One retry for transient Docker container startup failures.
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 60_000,
  globalTimeout: process.env.CI ? ciGlobalTimeoutMs : 0, // 20 min hard cap in CI
  reporter: [
    ["line"],
    [
      "json",
      { outputFile: "test-results-mock-llm-docker/results.json" },
    ],
    [
      "html",
      {
        outputFolder: "playwright-report-mock-llm-docker",
        open: "never",
      },
    ],
    ["./tests/e2e/mock-llm/reporters/done-marker-reporter.ts"],
  ],
  outputDir: "test-results-mock-llm-docker",
  use: {
    baseURL: INGRESS_URL,
    screenshot: "only-on-failure",
    trace: "on-first-retry",
    video: "on",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    // 1. Mock LLM server (Python, on the host)
    {
      command: `${MOCK_LLM_PYTHON} tests/e2e/mock-llm/scripts/mock-llm-server.py --port ${MOCK_LLM_PORT}`,
      url: MOCK_LLM_URL,
      timeout: 30_000,
      reuseExistingServer: !process.env.CI,
      stdout: "pipe",
      stderr: "pipe",
    },
    // 2. Docker container running the agent-canvas all-in-one image
    //
    // Uses --network host so the container shares the host's network:
    //   - The ingress port is available at localhost:<INGRESS_PORT>
    //   - The agent-server can reach the mock LLM at 127.0.0.1:<MOCK_LLM_PORT>
    //
    // The container is started with --rm for auto-cleanup. A named container
    // is used so the teardown can `docker stop` it reliably.
    //
    // Note: --network host is Linux-only. On macOS/Windows Docker Desktop,
    // use -p port mapping and set MOCK_LLM_AGENT_URL=http://host.docker.internal:<port>.
    {
      command: [
        // Stop any leftover container from a previous failed run
        `docker rm -f ${CONTAINER_NAME} 2>/dev/null;`,
        "exec docker run",
        "--rm",
        `--name ${CONTAINER_NAME}`,
        "--network host",
        // Mount the mock ACP server script so the agent-server inside
        // Docker can spawn it as an ACP subprocess.
        `-v ${MOCK_ACP_HOST_PATH}:${MOCK_ACP_CONTAINER_PATH}:ro`,
        // Mount skill test directories so the agent-server can access
        // repos and user skills created by the host-side test code.
        `-v ${SKILL_REPOS_HOST_DIR}:${SKILL_REPOS_CONTAINER_DIR}`,
        `-v ${USER_SKILLS_HOST_DIR}:${USER_SKILLS_CONTAINER_DIR}`,
        // Mount the folder-workspace test directory so the agent-server's
        // folder browser can list/navigate it inside the container.
        `-v ${FOLDER_WORKSPACE_HOST_DIR}:${FOLDER_WORKSPACE_CONTAINER_DIR}`,
        `-e PORT=${INGRESS_PORT}`,
        `-e SESSION_API_KEY=${sessionApiKey}`,
        `-e OH_SESSION_API_KEYS_0=${sessionApiKey}`,
        `-e PUBLIC_MODE_PORT=${PUBLIC_MODE_PORT}`,
        "-e VITE_DO_NOT_TRACK=1",
        "-e VITE_ENABLE_BROWSER_TOOLS=false",
        DOCKER_IMAGE,
      ].join(" "),
      // Probe the automation list endpoint through the ingress to ensure
      // the FULL stack (agent-server + automation backend + ingress) is
      // up before tests start. GET /api/automation/v1 returns 200 (empty
      // list) without auth — the automation backend does not enforce
      // session-key auth on the list endpoint.
      url: `http://localhost:${INGRESS_PORT}/api/automation/v1`,
      timeout: 180_000, // Docker pull + all services startup
      reuseExistingServer: !process.env.CI,
    },
  ],
  // globalTeardown stops the Docker container when Playwright exits.
  // Playwright sends SIGTERM to the webServer command, but `docker run`
  // with --rm handles cleanup automatically on termination.
});
