/**
 * REST client for DigitalOcean MARS ("hosted agents") sessions.
 *
 * Runs in the Electron main process, never the renderer: harness-api emits no
 * CORS headers, so a browser origin cannot call these endpoints at all, and
 * the bearer token must not cross the context bridge.
 *
 * Endpoint shapes are those served by harness-api's hand-rolled gorilla/mux
 * router (NOT grpc-gateway), so paths and the response envelopes below are the
 * contract — list wraps in `sessions`, single reads wrap in `session`, and
 * every error is `{error: {code, message}}`.
 */

export const DEFAULT_MARS_API_BASE_URL = "https://api.digitalocean.com";

/**
 * Port the guest's `openhands-agent-server` listens on inside the sandbox.
 * Within harness-api's allowed forwarding range (1024-65535).
 */
export const AGENT_SERVER_GUEST_PORT = 8000;

const SESSIONS_PATH = "/v2/agents/sessions";
const CONFIGS_PATH = "/v2/agents/configs";
const CONFIG_SESSIONS_SUBPATH = "/sessions";
const DEVICE_ID_HEADER = "X-Device-UUID";
const DEFAULT_TIMEOUT_MS = 15_000;
/**
 * Creating a session allocates a sandbox before responding, so it routinely
 * outlasts the read timeout that every other call is sized for.
 */
const CREATE_SESSION_TIMEOUT_MS = 60_000;

/** Terminal states — a session in one of these can never serve a tunnel. */
const TERMINAL_STATUSES = new Set([
  "SESSION_STATUS_DESTROYING",
  "SESSION_STATUS_DESTROYED",
  "SESSION_STATUS_FAILED",
]);

export const SESSION_STATUS_READY = "SESSION_STATUS_READY";
export const SESSION_STATUS_PAUSED = "SESSION_STATUS_PAUSED";

export function isTerminalSessionStatus(status) {
  return TERMINAL_STATUSES.has(status);
}

/**
 * The coding agent a config's manifest runs (`agent: openhands`), lowercased.
 * Manifests are written either flat or as `{apiVersion, kind, spec}`, and
 * `agent` may be a bare name or an object naming it; null when absent.
 *
 * @param {unknown} manifest
 * @returns {string | null}
 */
export function readManifestAgent(manifest) {
  if (!manifest || typeof manifest !== "object") return null;
  const agent = manifest.agent ?? manifest.spec?.agent;
  const name =
    typeof agent === "string" ? agent : (agent?.name ?? agent?.kind ?? null);
  return typeof name === "string" && name.trim()
    ? name.trim().toLowerCase()
    : null;
}

export class MarsApiError extends Error {
  constructor(message, { status = null, cause = null } = {}) {
    super(message);
    this.name = "MarsApiError";
    this.status = status;
    if (cause) this.cause = cause;
  }
}

function trimTrailingSlash(value) {
  return value.replace(/\/+$/, "");
}

/**
 * Build a fully-qualified MARS URL. Any path prefix on `baseUrl` is preserved
 * so non-production edges (`https://host/edge`) keep routing correctly, which
 * mirrors how doctl derives its endpoints from `--api-url`.
 */
export function buildMarsUrl(baseUrl, path, query = {}) {
  const url = new URL(`${trimTrailingSlash(baseUrl)}${path}`);
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

/**
 * Pull the human-readable message out of harness-api's error envelope. Falls
 * back to the status line so a proxy/edge error page (which is not JSON) still
 * produces something actionable rather than "undefined".
 */
async function readErrorMessage(response) {
  try {
    const body = await response.json();
    const message = body?.error?.message;
    if (typeof message === "string" && message) return message;
  } catch {
    // Non-JSON body (edge HTML, empty 502) — fall through to the status line.
  }
  return `${response.status} ${response.statusText}`.trim();
}

/**
 * @param {object} options
 * @param {() => Promise<string | null>} options.getToken Resolves the bearer
 *   token at call time so a rotated or re-authenticated credential is picked
 *   up without rebuilding the client.
 * @param {string} [options.baseUrl]
 * @param {string | null} [options.deviceId]
 * @param {typeof globalThis.fetch} [options.fetchImpl] Injectable for tests.
 */
export function createMarsApiClient({
  getToken,
  baseUrl = DEFAULT_MARS_API_BASE_URL,
  deviceId = null,
  fetchImpl = globalThis.fetch,
}) {
  async function request(method, path, { query, timeoutMs, body } = {}) {
    const token = await getToken();
    if (!token) {
      throw new MarsApiError("Not signed in to DigitalOcean.", { status: 401 });
    }

    const headers = { Authorization: `Bearer ${token}` };
    if (deviceId) headers[DEVICE_ID_HEADER] = deviceId;
    if (body !== undefined) headers["Content-Type"] = "application/json";

    let response;
    try {
      response = await fetchImpl(buildMarsUrl(baseUrl, path, query), {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs ?? DEFAULT_TIMEOUT_MS),
      });
    } catch (error) {
      throw new MarsApiError(
        `Could not reach DigitalOcean: ${error instanceof Error ? error.message : String(error)}`,
        { cause: error },
      );
    }

    if (!response.ok) {
      throw new MarsApiError(await readErrorMessage(response), {
        status: response.status,
      });
    }

    // 204 on pause/resume/delete — no body to parse.
    if (response.status === 204) return null;
    return response.json();
  }

  return {
    /** `{sessions, next_page_token}`; results are always team-scoped. */
    async listSessions({ pageSize, pageToken, status } = {}) {
      const body = await request("GET", SESSIONS_PATH, {
        query: { page_size: pageSize, page_token: pageToken, status },
      });
      return {
        sessions: Array.isArray(body?.sessions) ? body.sessions : [],
        nextPageToken: body?.next_page_token || null,
      };
    },

    async getSession(sessionId) {
      const body = await request("GET", `${SESSIONS_PATH}/${sessionId}`);
      return body?.session ?? null;
    },

    /**
     * Agent Configs are the durable agent definitions (one config, many
     * sessions). List items are the lightweight `AgentConfigSummary`
     * projection, so they carry no manifest body.
     */
    async listAgentConfigs({ pageSize, pageToken } = {}) {
      const body = await request("GET", CONFIGS_PATH, {
        query: { page_size: pageSize, page_token: pageToken },
      });
      return {
        configs: Array.isArray(body?.configs) ? body.configs : [],
        nextPageToken: body?.next_page_token || null,
      };
    },

    /** Full config, including the parsed `manifest` the list omits. */
    async getAgentConfig(configId) {
      const body = await request("GET", `${CONFIGS_PATH}/${configId}`);
      return body?.config ?? null;
    },

    /**
     * Sessions belonging to one config. Same Session shape and pagination as
     * `listSessions`; a config that exists but has no sessions is an empty
     * 200, so an empty list here does NOT mean the config is gone.
     */
    async listConfigSessions(configId, { pageSize, pageToken, status } = {}) {
      const body = await request(
        "GET",
        `${CONFIGS_PATH}/${configId}${CONFIG_SESSIONS_SUBPATH}`,
        { query: { page_size: pageSize, page_token: pageToken, status } },
      );
      return {
        sessions: Array.isArray(body?.sessions) ? body.sessions : [],
        nextPageToken: body?.next_page_token || null,
      };
    },

    /**
     * Launch a new session from an existing Agent Config. The config already
     * holds the manifest and its resolved credentials, so this carries no
     * spec of its own — which is what makes "run this agent" a single call.
     * The new session starts PROVISIONING, not READY.
     */
    async createSessionFromConfig(configId, name) {
      const body = await request("POST", SESSIONS_PATH, {
        body: { name, config_id: configId },
        timeoutMs: CREATE_SESSION_TIMEOUT_MS,
      });
      return body?.session ?? null;
    },

    async pauseSession(sessionId) {
      await request("POST", `${SESSIONS_PATH}/${sessionId}/pause`);
    },

    async resumeSession(sessionId) {
      await request("POST", `${SESSIONS_PATH}/${sessionId}/resume`);
    },

    /**
     * Cheapest authenticated call that proves the token is valid AND that the
     * caller actually has MARS access. A well-formed token can still be
     * refused by the `mars_preview` / `mars_access_ric1` feature flippers, so
     * validating the string shape alone would report success too early.
     */
    async verifyAccess() {
      await request("GET", SESSIONS_PATH, { query: { page_size: 1 } });
    },
  };
}
