/**
 * REST client for Maritime (https://maritime.sh) agents.
 *
 * Runs in a trusted Node process (the Electron main process, or the Agent
 * Canvas server in the web build), never the renderer: the Maritime API key
 * can create, wake and delete machines on the user's account, so it must not
 * reach page JavaScript.
 *
 * A Maritime agent is a Firecracker micro-VM running a container image. Every
 * agent with an exposed port gets its own HTTPS and WebSocket hostname,
 * `https://<agent-id>.mtm.sh/`, reported as `webUrl`, so Canvas reaches the
 * agent-server inside it directly, with no tunnel. Maritime sleeps an idle
 * agent with its memory kept and wakes it on the next HTTP request.
 *
 * Field names are those of the Maritime API (camelCase on the wire). Errors
 * come back as `{detail}` (a string, or FastAPI's validation list) and some
 * carry a `code` such as `seat_limit`.
 */

export const DEFAULT_MARITIME_API_BASE_URL = "https://api.maritime.sh";

/**
 * The OpenHands agent-server image built for Maritime: the published
 * `ghcr.io/openhands/agent-server:<version>-python` image with a launcher that
 * binds `$PORT`, the port Maritime routes the agent's hostname to. Tags follow
 * the agent-server version.
 */
export const DEFAULT_MARITIME_IMAGE_REPOSITORY =
  "ghcr.io/maritime-sh/openhands-agent-server";

/**
 * Tags the agents Canvas creates, so the list shows only those and never the
 * user's other Maritime agents. Maritime filters on it server-side.
 */
export const OPENHANDS_EXTERNAL_ID = "openhands-agent-canvas";

/**
 * The port Maritime publishes for a custom image. It forwards it to `$PORT`
 * inside the VM, which Maritime sets and the image's launcher binds.
 */
export const MARITIME_PUBLIC_PORT = 8080;

/**
 * Seconds without HTTP traffic before Maritime puts an agent to sleep. Open
 * WebSocket frames do not count as traffic, a busy CPU does, so a working
 * agent stays up while an idle one stops costing an awake slot. Sleep keeps
 * memory, and the next request wakes it in about a second.
 */
export const DEFAULT_IDLE_TTL_SECONDS = 30 * 60;

export const MARITIME_STATUS_ACTIVE = "active";
export const MARITIME_STATUS_SLEEPING = "sleeping";
export const MARITIME_STATUS_DEPLOYING = "deploying";
export const MARITIME_STATUS_ERROR = "error";
export const MARITIME_STATUS_STOPPED = "stopped";

const AGENTS_PATH = "/api/agents";
const LIMITS_PATH = "/api/v1/account/limits";
const DEFAULT_TIMEOUT_MS = 15_000;
/** Create builds the VM's root filesystem from the image before answering. */
const CREATE_TIMEOUT_MS = 120_000;
/** Start and restart answer once the VM is back. */
const LIFECYCLE_TIMEOUT_MS = 90_000;

/** Prefix of every Maritime API key. */
const API_KEY_PREFIX = "mk_";
/** Shape of an agent-server image tag, so a config value cannot inject anything. */
const IMAGE_TAG = /^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/;

export class MaritimeApiError extends Error {
  /**
   * @param {string} message
   * @param {{ status?: number | null, code?: string | null, cause?: unknown }} [options]
   */
  constructor(message, { status = null, code = null, cause = null } = {}) {
    super(message);
    this.name = "MaritimeApiError";
    this.status = status;
    this.code = code;
    if (cause) this.cause = cause;
  }
}

/**
 * Fast typo guard before a key is sent anywhere. It cannot tell whether the
 * key is live or has the scopes Canvas needs; the API answers that.
 */
export function isValidApiKeyFormat(key) {
  const trimmed = typeof key === "string" ? key.trim() : "";
  return trimmed.startsWith(API_KEY_PREFIX) && trimmed.length > 16;
}

/** `ghcr.io/maritime-sh/openhands-agent-server:<tag>`, checked. */
export function buildImageName(repository, tag) {
  if (typeof tag !== "string" || !IMAGE_TAG.test(tag)) {
    throw new MaritimeApiError(`Invalid agent-server image tag: ${tag}`);
  }
  return `${repository}:${tag}`;
}

/**
 * vCPUs for a memory size: one per 2 GB, from one to four. Maritime takes any
 * value; this keeps the picker to one choice.
 */
export function vcpusForMemory(memMb) {
  return Math.min(4, Math.max(1, Math.round(memMb / 2048)));
}

/**
 * The `POST /api/agents` body for an OpenHands agent. Built here, not taken
 * from the renderer, so the renderer can only ever create agent-servers from
 * the configured image.
 *
 * `publicWeb` lets a browser reach the agent's hostname without a Maritime
 * session cookie, which CORS preflights could never carry. The agent-server's
 * own session API key is what guards it, as on every remote backend.
 *
 * @param {object} input
 * @param {string} input.name
 * @param {string} input.imageName
 * @param {number} input.memMb
 * @param {string} input.sessionApiKey Sent as `OH_SESSION_API_KEYS_0`
 * @param {string} input.secretKey Sent as `OH_SECRET_KEY`; encrypts the
 *   agent-server's stored settings and secrets
 * @param {string[]} [input.corsOrigins] Non-loopback Canvas origins the
 *   agent-server must accept; loopback origins are always allowed
 * @param {number} [input.idleTtlSeconds]
 */
export function buildOpenHandsAgentRequest({
  name,
  imageName,
  memMb,
  sessionApiKey,
  secretKey,
  corsOrigins = [],
  idleTtlSeconds = DEFAULT_IDLE_TTL_SECONDS,
}) {
  const env = [
    { key: "OH_SESSION_API_KEYS_0", value: sessionApiKey, isSecret: true },
    { key: "OH_SECRET_KEY", value: secretKey, isSecret: true },
    ...corsOrigins.map((origin, index) => ({
      key: `OH_ALLOW_CORS_ORIGINS_${index}`,
      value: origin,
      isSecret: false,
    })),
  ];
  return {
    name,
    description: "OpenHands agent server, created by OpenHands Agent Canvas.",
    externalId: OPENHANDS_EXTERNAL_ID,
    framework: "custom",
    imageName,
    exposedPort: MARITIME_PUBLIC_PORT,
    publicWeb: true,
    hasWebUi: false,
    healthCheckPath: "/health",
    memMb,
    vcpus: vcpusForMemory(memMb),
    idleTtlSeconds,
    initialEnvVars: env,
  };
}

/**
 * What the renderer sees of an agent. The Maritime record carries nothing
 * secret, but only these fields are part of the bridge contract.
 */
export function toAgentSummary(agent) {
  return {
    id: agent.id,
    name: agent.name ?? agent.id,
    status: agent.status ?? null,
    webUrl: agent.webUrl ?? null,
    memMb: agent.memMb ?? null,
    vcpus: agent.vcpus ?? null,
    createdAt: agent.createdAt ?? null,
    lastActiveAt: agent.lastActiveAt ?? null,
  };
}

/** The agent-server base URL for an agent: its hostname, no trailing slash. */
export function agentServerHost(agent) {
  const url = agent?.webUrl;
  return typeof url === "string" && url ? url.replace(/\/+$/, "") : null;
}

/** Readable message from a FastAPI error body. */
function readDetail(body) {
  const detail = body?.detail;
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (typeof item?.msg === "string" ? item.msg : null))
      .filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  if (typeof body?.error === "string") return body.error;
  return null;
}

/**
 * The message a failed call surfaces. It has to stand on its own: Electron
 * IPC keeps only an error's message, not its status or code. Maritime's own
 * `detail` is already written for people (plan limits, name clashes), so it
 * is kept wherever it is the better explanation.
 */
export function describeMaritimeFailure(status, detail) {
  switch (status) {
    case 401:
      return "Maritime did not accept this API key. Check that it is current, or create a new one.";
    case 403:
      return "This Maritime API key is missing a permission. Create a key with the provision, deploy and secrets scopes.";
    case 402:
      return detail ?? "Your Maritime plan has no room for another agent.";
    case 429:
      return detail
        ? `Maritime limit reached: ${detail}`
        : "Too many agents are awake on your Maritime plan. Put one to sleep, or wait a moment.";
    default:
      return detail ?? `Maritime answered with status ${status}.`;
  }
}

function trimTrailingSlash(value) {
  return value.replace(/\/+$/, "");
}

/**
 * @param {object} options
 * @param {string} [options.baseUrl]
 * @param {() => Promise<string | null> | string | null} options.getApiKey
 * @param {typeof fetch} [options.fetchImpl]
 */
export function createMaritimeApiClient({
  baseUrl = DEFAULT_MARITIME_API_BASE_URL,
  getApiKey,
  fetchImpl = (...args) => fetch(...args),
}) {
  const root = trimTrailingSlash(baseUrl);

  async function request(
    method,
    path,
    { body, timeoutMs = DEFAULT_TIMEOUT_MS, query } = {},
  ) {
    const apiKey = await getApiKey();
    if (!apiKey) {
      throw new MaritimeApiError("Sign in to Maritime first.", { status: 401 });
    }
    const url = new URL(`${root}${path}`);
    for (const [name, value] of Object.entries(query ?? {})) {
      if (value !== undefined && value !== null) {
        url.searchParams.set(name, String(value));
      }
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let response;
    try {
      response = await fetchImpl(url.toString(), {
        method,
        headers: {
          Authorization: `Bearer ${apiKey}`,
          Accept: "application/json",
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      const timedOut = controller.signal.aborted;
      throw new MaritimeApiError(
        timedOut
          ? "Maritime did not answer in time."
          : "Could not reach Maritime.",
        { cause: error },
      );
    } finally {
      clearTimeout(timer);
    }

    if (response.status === 204) return null;
    let parsed = null;
    const text = await response.text();
    if (text) {
      try {
        parsed = JSON.parse(text);
      } catch {
        // An edge error page; the status line says enough.
      }
    }
    if (!response.ok) {
      throw new MaritimeApiError(
        describeMaritimeFailure(response.status, readDetail(parsed)),
        {
          status: response.status,
          code: typeof parsed?.code === "string" ? parsed.code : null,
        },
      );
    }
    return parsed;
  }

  const agentPath = (id) => `${AGENTS_PATH}/${encodeURIComponent(id)}`;

  return {
    /** Plan, machine counts and per-agent size range. Also proves the key works. */
    getAccountLimits: () => request("GET", LIMITS_PATH),

    /** Agents tagged with `externalId`; the API answers a bare array. */
    async listAgents({ externalId = OPENHANDS_EXTERNAL_ID } = {}) {
      const agents = await request("GET", AGENTS_PATH, {
        query: { externalId },
      });
      return Array.isArray(agents) ? agents : [];
    },

    getAgent: (id) => request("GET", agentPath(id)),

    createAgent: (body) =>
      request("POST", AGENTS_PATH, { body, timeoutMs: CREATE_TIMEOUT_MS }),

    startAgent: (id) =>
      request("POST", `${agentPath(id)}/start`, {
        timeoutMs: LIFECYCLE_TIMEOUT_MS,
      }),

    sleepAgent: (id) =>
      request("POST", `${agentPath(id)}/sleep`, {
        timeoutMs: LIFECYCLE_TIMEOUT_MS,
      }),

    restartAgent: (id) =>
      request("POST", `${agentPath(id)}/restart`, {
        timeoutMs: LIFECYCLE_TIMEOUT_MS,
      }),

    deleteAgent: (id) => request("DELETE", agentPath(id)),

    /** Upserts one env var. Needs the `secrets` scope. */
    setEnvVar: (id, key, value, { isSecret = true } = {}) =>
      request("POST", `${agentPath(id)}/env`, {
        body: { key, value, isSecret },
      }),
  };
}
