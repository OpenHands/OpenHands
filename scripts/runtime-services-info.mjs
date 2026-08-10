/**
 * Single source of truth for the `<RUNTIME_SERVICES>` block.
 *
 * Builds a structured description of the services that are reachable from
 * inside the agent's sandbox. Agent Canvas backend-serving processes attach it
 * to `/server_info.runtime_services`; the frontend renders that backend value
 * into `AgentContext.system_message_suffix`, so the agent sees a
 * `<RUNTIME_SERVICES>` block listing what's available without having to probe.
 *
 * Two callers share this one definition:
 *   - the dev launchers (scripts/dev-*.mjs), which know the stack as a set of
 *     ports and pass the result to ingress/static-server for `/server_info`;
 *   - docker/entrypoint.sh, which runs this file as a CLI (see the bottom of
 *     this module) because in a container the URLs are *runtime* config — the
 *     ports and base URLs are overridable at `docker run` and therefore cannot
 *     be baked into the image at build time. The JSON it prints is passed to
 *     scripts/static-server.mjs and exposed through `/server_info`.
 *
 * URLs are written from the *agent's* point of view (i.e. as the agent should
 * curl/fetch them from inside its sandbox), which is deliberately not the
 * browser's point of view.
 */

import process from "node:process";
import { pathToFileURL } from "node:url";

/**
 * @param {object} options
 * @param {string} [options.mode] - Human-readable mode label (e.g. "dev:safe").
 * @param {string} [options.agentHostAlias="localhost"] - Hostname the agent
 *   uses to reach host-side services (ingress, frontend, port-derived
 *   automation). Also surfaced as `agent_host_alias`.
 * @param {number} [options.agentServerPort] - Port the agent-server listens on.
 *   Used to derive the agent_server URL when `agentServerUrl` is not given.
 * @param {string} [options.agentServerUrl] - Explicit agent_server URL, from
 *   the agent's POV. Takes precedence over `agentServerPort`; used by the
 *   Docker image, which serves over `127.0.0.1` to avoid IPv6 loopback issues
 *   and honors an overridable `AGENT_SERVER_URL`. One of `agentServerUrl` /
 *   `agentServerPort` is required (else the URL would be `:undefined`).
 * @param {number} [options.ingressPort] - Ingress port (omit if no ingress).
 * @param {number} [options.frontendPort] - Frontend port (Vite dev server
 *   or static-file server). Omit if no frontend is exposed.
 * @param {number} [options.vitePort] - Deprecated alias for `frontendPort`,
 *   accepted for backward compat with older launchers. Remove after one release.
 * @param {"vite"|"static"} [options.frontendKind="vite"] - Whether the
 *   frontend port hosts Vite or a static build. Only affects the description.
 * @param {object} [options.automation] - Automation backend info. Skipped
 *   entirely unless `.url` or `.port` is provided, so passing `{}` is safe.
 * @param {string} [options.automation.url] - Explicit automation base URL, from
 *   the agent's POV. Takes precedence over `.port`; used by the Docker image to
 *   honor an overridable `AUTOMATION_BASE_URL`.
 * @param {number} [options.automation.port] - Automation backend port (used to
 *   derive the base URL when `.url` is not given).
 * @param {string} [options.automation.apiPrefix="/api/automation"] - Path
 *   prefix all automation routes are mounted under.
 * @param {string} [options.automation.authEnvVar="OPENHANDS_AUTOMATION_API_KEY"]
 *   - Env var holding the API key.
 * @returns {object} A JSON-serializable runtime services info object.
 */
export function buildRuntimeServicesInfo(options) {
  const {
    mode,
    agentHostAlias = "localhost",
    agentServerPort,
    agentServerUrl,
    ingressPort,
    // Accept legacy `vitePort` for one release so external callers keep working.
    vitePort,
    frontendPort = vitePort,
    frontendKind = "vite",
    automation,
  } = options;

  // Prefer an explicit URL (containers reach the agent-server over a specific
  // host/scheme), else derive it from the port. From the agent's POV the
  // agent-server it's *inside* is on the loopback host, regardless of where
  // the host machine is.
  const agentServerUrlResolved =
    agentServerUrl ??
    (agentServerPort != null ? `http://localhost:${agentServerPort}` : null);
  if (!agentServerUrlResolved) {
    // Without this the URL becomes `http://localhost:undefined` and ends up
    // verbatim in the agent's system prompt, which is worse than failing fast.
    throw new Error(
      "buildRuntimeServicesInfo: agentServerPort or agentServerUrl is required " +
        "(otherwise the agent_server URL would be `http://localhost:undefined`).",
    );
  }

  const services = {
    agent_server: {
      description:
        "The OpenHands Agent Server this agent is running inside. " +
        "Tool calls (terminal, file_editor, browser, etc.) execute here.",
      url_from_agent: agentServerUrlResolved,
    },
  };

  if (ingressPort !== undefined) {
    services.ingress = {
      description:
        "Unified entry point. Routes /api/automation/* to the automation " +
        "backend, /api/* and /sockets to the agent-server, and /* to the " +
        "frontend.",
      url_from_agent: `http://${agentHostAlias}:${ingressPort}`,
    };
  }

  if (frontendPort !== undefined) {
    services.frontend = {
      kind: frontendKind,
      description:
        frontendKind === "static"
          ? "Static-file server hosting the agent-canvas production build."
          : "Vite dev server hosting the agent-canvas frontend.",
      url_from_agent: `http://${agentHostAlias}:${frontendPort}`,
    };
  }

  // Prefer an explicit base URL, else derive from the port. Require one of the
  // two so we don't bake `:undefined` into the URL when the caller passes
  // `automation: {}`.
  const automationBaseUrl =
    automation?.url ??
    (automation?.port != null
      ? `http://${agentHostAlias}:${automation.port}`
      : null);
  if (automationBaseUrl) {
    const apiPrefix = automation.apiPrefix ?? "/api/automation";
    const authEnvVar = automation.authEnvVar ?? "OPENHANDS_AUTOMATION_API_KEY";
    services.automation = {
      description:
        "OpenHands Automations service. All routes are mounted under " +
        `'${apiPrefix}'. Authenticate with header ` +
        `'X-Session-API-Key: $${authEnvVar}'.`,
      url_from_agent: automationBaseUrl,
      api_prefix: apiPrefix,
      docs_url: `${automationBaseUrl}${apiPrefix}/docs`,
      openapi_url: `${automationBaseUrl}${apiPrefix}/openapi.json`,
      auth_env_var: authEnvVar,
    };
  }

  return {
    mode,
    agent_host_alias: agentHostAlias,
    services,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI — used by docker/entrypoint.sh to emit the JSON at container startup.
// ─────────────────────────────────────────────────────────────────────────────

export function parseArgs(argv) {
  const options = { automation: {} };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    switch (flag) {
      case "--mode":
        options.mode = argv[++i];
        break;
      case "--agent-host-alias":
        options.agentHostAlias = argv[++i];
        break;
      case "--agent-server-url":
        options.agentServerUrl = argv[++i] || undefined;
        break;
      case "--automation-url":
        options.automation.url = argv[++i] || undefined;
        break;
      case "--automation-api-prefix":
        options.automation.apiPrefix = argv[++i];
        break;
      case "--automation-auth-env":
        options.automation.authEnvVar = argv[++i];
        break;
      default:
        throw new Error(`Unknown flag: ${flag}`);
    }
  }
  // Omit the automation entry entirely when no URL was supplied, rather than
  // advertising a backend the agent cannot reach.
  if (!options.automation.url) delete options.automation;
  return options;
}

const isMainModule =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMainModule) {
  try {
    const options = parseArgs(process.argv.slice(2));
    process.stdout.write(JSON.stringify(buildRuntimeServicesInfo(options)));
  } catch (err) {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  }
}
