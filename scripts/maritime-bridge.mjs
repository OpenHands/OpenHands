/**
 * Trusted-process service behind the Maritime provider (Add Backend →
 * Providers → Maritime).
 *
 * Wires the Maritime REST client (scripts/maritime-api.mjs) and the Maritime
 * credential store into Electron's IPC layer; the Agent Canvas server exposes
 * the same methods to the web build (scripts/maritime-web-bridge.mjs). The
 * renderer reaches it only as `window.maritimeBridge`.
 *
 * Each OpenHands agent on Maritime is one Maritime agent (a micro-VM) running
 * the agent-server image, reachable at its own HTTPS hostname. Opening one
 * hands the renderer that hostname and the agent-server's session API key,
 * which it registers as an ordinary local backend. The Maritime API key never
 * leaves this process; the session key does, because the browser talks to
 * the agent-server directly, exactly as it does for any remote backend.
 */

import { randomBytes } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_MARITIME_API_BASE_URL,
  DEFAULT_MARITIME_IMAGE_REPOSITORY,
  MARITIME_STATUS_ACTIVE,
  MARITIME_STATUS_ERROR,
  MARITIME_STATUS_SLEEPING,
  MARITIME_STATUS_STOPPED,
  MaritimeApiError,
  agentServerHost,
  buildImageName,
  buildOpenHandsAgentRequest,
  createMaritimeApiClient,
  isValidApiKeyFormat,
  toAgentSummary,
} from "./maritime-api.mjs";
import { createMaritimeCredentialStore } from "./maritime-credentials.mjs";

export const MARITIME_IPC = {
  getAuthState: "maritime:getAuthState",
  saveApiKey: "maritime:saveApiKey",
  signOut: "maritime:signOut",
  getAccount: "maritime:getAccount",
  listAgents: "maritime:listAgents",
  createAgent: "maritime:createAgent",
  openAgent: "maritime:openAgent",
  resetAccessKey: "maritime:resetAccessKey",
  sleepAgent: "maritime:sleepAgent",
  deleteAgent: "maritime:deleteAgent",
};

/** First boot builds the VM's root filesystem from the image. */
export const CREATE_READY_TIMEOUT_MS = 10 * 60_000;
/** Waking from sleep takes about a second; a cold start takes longer. */
export const WAKE_READY_TIMEOUT_MS = 3 * 60_000;
const POLL_INTERVAL_MS = 2_000;

/** Maritime's self-serve memory range; the account's own range narrows it. */
const MIN_MEM_MB = 512;
const MAX_MEM_MB = 8192;
/** Letters, digits, dots, dashes and underscores; alphanumeric at both ends. */
const AGENT_NAME = /^[A-Za-z0-9]([A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$/;

export const MISSING_ACCESS_KEY_CODE = "missing_access_key";

/**
 * Shipped Maritime defaults from `config/defaults.json`, plus the
 * agent-server version Canvas pins, which picks the image tag.
 *
 * @returns {{ apiBaseUrl?: string, imageRepository?: string, agentServerVersion?: string }}
 */
export function readMaritimeDefaults() {
  try {
    const defaults = JSON.parse(
      readFileSync(
        fileURLToPath(new URL("../config/defaults.json", import.meta.url)),
        "utf-8",
      ),
    );
    return {
      ...(defaults.maritime ?? {}),
      agentServerVersion: defaults.versions?.agentServer,
    };
  } catch {
    return {};
  }
}

/**
 * Env wins over shipped defaults, so a staging API or a private image can be
 * used without rebuilding.
 *
 * @param {Record<string, string | undefined>} [env]
 * @param {ReturnType<typeof readMaritimeDefaults>} [defaults]
 */
export function readMaritimeConfig(
  env = process.env,
  defaults = readMaritimeDefaults(),
) {
  const repository =
    env.MARITIME_IMAGE_REPOSITORY ||
    defaults.imageRepository ||
    DEFAULT_MARITIME_IMAGE_REPOSITORY;
  const tag = env.MARITIME_IMAGE_TAG || defaults.agentServerVersion;
  return {
    apiBaseUrl:
      env.MARITIME_API_BASE_URL ||
      defaults.apiBaseUrl ||
      DEFAULT_MARITIME_API_BASE_URL,
    imageName: tag ? buildImageName(repository, tag) : `${repository}:latest`,
  };
}

/** A random secret for the agent-server, 256 bits as hex. */
function newSecret() {
  return randomBytes(32).toString("hex");
}

/**
 * The origin to add to the agent-server's CORS list, or null. Loopback
 * origins (the desktop app, `npx @openhands/agent-canvas`) are always allowed
 * by the agent-server, so only a Canvas served elsewhere needs one.
 */
export function corsOriginFor(origin) {
  if (typeof origin !== "string" || !origin) return null;
  let url;
  try {
    url = new URL(origin);
  } catch {
    return null;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  const host = url.hostname.replace(/^\[|\]$/g, "");
  if (
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host === "::1" ||
    /^127\./.test(host)
  ) {
    return null;
  }
  return url.origin;
}

const sleep = (ms) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

/**
 * @param {object} [options]
 * @param {ReturnType<typeof createMaritimeCredentialStore>} [options.credentials] Override for tests
 * @param {ReturnType<typeof createMaritimeApiClient>} [options.api] Override for tests
 * @param {string | null} [options.userDataPath]
 * @param {import("electron").SafeStorage | null} [options.safeStorage]
 * @param {import("./mars-secret-store.mjs").SecretStore | null} [options.secrets]
 * @param {Record<string, string | undefined>} [options.env]
 * @param {typeof fetch} [options.fetchImpl]
 * @param {(ms: number) => Promise<void>} [options.wait] Override for tests
 * @param {() => number} [options.now] Override for tests
 */
export function createMaritimeBridge({
  credentials,
  api,
  userDataPath = null,
  safeStorage = null,
  secrets = null,
  env = process.env,
  fetchImpl,
  wait = sleep,
  now = Date.now,
} = {}) {
  const config = readMaritimeConfig(env);
  const store =
    credentials ??
    createMaritimeCredentialStore({ userDataPath, safeStorage, secrets });
  const clientFor = (getApiKey) =>
    createMaritimeApiClient({
      baseUrl: config.apiBaseUrl,
      getApiKey,
      ...(fetchImpl ? { fetchImpl } : {}),
    });
  const client = api ?? clientFor(() => store.getApiKey());

  function authState() {
    return {
      isSignedIn: store.getApiKey() !== null,
      isSaved: store.isSaved,
      storage: store.storage,
    };
  }

  function requireSignedIn() {
    if (store.getApiKey() === null) {
      throw new MaritimeApiError("Sign in to Maritime first.", { status: 401 });
    }
  }

  function summarize(agent) {
    return {
      ...toAgentSummary(agent),
      hasAccessKey: store.getSessionKey(agent.id) !== null,
    };
  }

  /**
   * Poll until Maritime reports the agent active. A sleeping or stopped
   * agent is started first. Error is final: Maritime keeps a failed deploy
   * failed until it is redeployed.
   */
  async function waitUntilActive(agentId, timeoutMs) {
    const deadline = now() + timeoutMs;
    let agent = await client.getAgent(agentId);
    let started = false;
    for (;;) {
      if (agent?.status === MARITIME_STATUS_ACTIVE && agentServerHost(agent)) {
        return agent;
      }
      if (agent?.status === MARITIME_STATUS_ERROR) {
        throw new MaritimeApiError(
          "Maritime could not start this agent. Open it on maritime.sh to see why.",
          { code: "agent_error" },
        );
      }
      if (
        !started &&
        (agent?.status === "sleeping" || agent?.status === "stopped")
      ) {
        started = true;
        agent = (await client.startAgent(agentId)) ?? agent;
        continue;
      }
      if (now() >= deadline) {
        throw new MaritimeApiError(
          "The agent did not come up in time. Try again in a minute.",
          { code: "timeout" },
        );
      }
      await wait(POLL_INTERVAL_MS);
      agent = await client.getAgent(agentId);
    }
  }

  function connection(agent, sessionApiKey) {
    return {
      agentId: agent.id,
      name: agent.name ?? agent.id,
      host: agentServerHost(agent),
      sessionApiKey,
    };
  }

  return {
    getAuthState: authState,

    /**
     * A well-formed key can still be revoked or belong to nobody, so it is
     * kept only once Maritime has accepted it.
     */
    async saveApiKey({ apiKey } = {}) {
      const trimmed = typeof apiKey === "string" ? apiKey.trim() : "";
      if (!isValidApiKeyFormat(trimmed)) {
        throw new MaritimeApiError(
          "That does not look like a Maritime API key (it starts with mk_).",
          { status: 400 },
        );
      }
      await clientFor(() => trimmed).getAccountLimits();
      store.saveApiKey(trimmed);
      return authState();
    },

    signOut() {
      store.forgetApiKey();
      return authState();
    },

    /** The plan and the memory range a new agent may use. */
    async getAccount() {
      requireSignedIn();
      const limits = await client.getAccountLimits();
      return {
        plan: limits.plan ?? null,
        machines: limits.machines ?? 0,
        includedMachines: limits.includedMachines ?? null,
        awakeMachines: limits.awakeMachines ?? 0,
        maxAwakeMachines: limits.maxAwakeMachines ?? null,
        minMemMb: Math.max(MIN_MEM_MB, limits.minMemMb ?? MIN_MEM_MB),
        maxMemMb: Math.min(MAX_MEM_MB, limits.maxMemMb ?? MAX_MEM_MB),
      };
    },

    async listAgents() {
      requireSignedIn();
      const agents = await client.listAgents();
      return agents.filter((agent) => agent?.id).map(summarize);
    },

    /**
     * Create an OpenHands agent-server on Maritime and wait until it runs.
     * The body is built here so the renderer can only ever choose a name, a
     * memory size and its own origin.
     */
    async createAgent({ name, memMb, origin } = {}) {
      requireSignedIn();
      const trimmedName = typeof name === "string" ? name.trim() : "";
      if (!AGENT_NAME.test(trimmedName)) {
        throw new MaritimeApiError(
          "Use 1 to 64 letters, digits, dots, dashes or underscores.",
          { status: 400 },
        );
      }
      if (
        !Number.isInteger(memMb) ||
        memMb < MIN_MEM_MB ||
        memMb > MAX_MEM_MB
      ) {
        throw new MaritimeApiError("Choose a memory size.", { status: 400 });
      }
      const sessionApiKey = newSecret();
      const corsOrigin = corsOriginFor(origin);
      const created = await client.createAgent(
        buildOpenHandsAgentRequest({
          name: trimmedName,
          imageName: config.imageName,
          memMb,
          sessionApiKey,
          secretKey: newSecret(),
          corsOrigins: corsOrigin ? [corsOrigin] : [],
        }),
      );
      if (!created?.id) {
        throw new MaritimeApiError("Maritime returned no agent.");
      }
      // Saved before waiting, so a slow first boot that times out can still
      // be opened later from the list.
      store.saveSessionKey(created.id, sessionApiKey);
      const agent = await waitUntilActive(created.id, CREATE_READY_TIMEOUT_MS);
      return connection(agent, sessionApiKey);
    },

    /**
     * Wake the agent if it sleeps and return what the renderer registers.
     * Without a stored session key (created on another machine, or the store
     * was lost) this refuses with MISSING_ACCESS_KEY_CODE; the UI then offers
     * `resetAccessKey`.
     */
    async openAgent(agentId) {
      requireSignedIn();
      const sessionApiKey = store.getSessionKey(agentId);
      if (!sessionApiKey) {
        throw new MaritimeApiError(
          "This device has no access key for that agent.",
          { code: MISSING_ACCESS_KEY_CODE },
        );
      }
      const agent = await waitUntilActive(agentId, WAKE_READY_TIMEOUT_MS);
      return connection(agent, sessionApiKey);
    },

    /**
     * Give the agent-server a new session key and restart it so it reads it.
     * Conversations and settings are kept (they live on the agent's
     * persistent disk); work running at that moment stops.
     */
    async resetAccessKey(agentId) {
      requireSignedIn();
      const sessionApiKey = newSecret();
      await client.setEnvVar(agentId, "OH_SESSION_API_KEYS_0", sessionApiKey, {
        isSecret: true,
      });
      store.saveSessionKey(agentId, sessionApiKey);
      await client.restartAgent(agentId);
      const agent = await waitUntilActive(agentId, WAKE_READY_TIMEOUT_MS);
      return connection(agent, sessionApiKey);
    },

    async sleepAgent(agentId) {
      requireSignedIn();
      try {
        return summarize(await client.sleepAgent(agentId));
      } catch (error) {
        // A sleep that races another one (a second click, Maritime's own idle
        // sleep) fails at Maritime with a bare 500. The agent is asleep either
        // way, which is what the user asked for.
        const current = await client.getAgent(agentId).catch(() => null);
        if (
          current?.status === MARITIME_STATUS_SLEEPING ||
          current?.status === MARITIME_STATUS_STOPPED
        ) {
          return summarize(current);
        }
        throw error;
      }
    },

    async deleteAgent(agentId) {
      requireSignedIn();
      await client.deleteAgent(agentId);
      store.forgetSessionKey(agentId);
    },

    /** @param {import("electron").IpcMain} ipcMain */
    registerIpc(ipcMain) {
      for (const [method, channel] of Object.entries(MARITIME_IPC)) {
        ipcMain.handle(channel, (_event, ...args) => this[method](...args));
      }
    },
  };
}
