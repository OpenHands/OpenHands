/**
 * Main-process service behind the DigitalOcean Managed Agents screen.
 *
 * Wires scripts/tunnel-registry.mjs (MARSOHS-1428/1429), the MARS REST client,
 * and the DigitalOcean credential store into Electron's IPC layer. Channel
 * names mirror the teammate fork's `window.marsBridge` surface so the two
 * integrations stay easy to reconcile.
 *
 * Tokens never cross the context bridge: the renderer asks for a tunnel by
 * session id alone, and this process sources the bearer token from the
 * active stored connection. harness-api sends no CORS headers, so every MARS
 * call has to happen here rather than in the renderer anyway.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createTunnelRegistry } from "./tunnel-registry.mjs";
import { ensureSessionAwake } from "./mars-session.mjs";
import {
  AGENT_SERVER_GUEST_PORT,
  DEFAULT_MARS_API_BASE_URL,
  MarsApiError,
  createMarsApiClient,
  readManifestAgent,
} from "./mars-api.mjs";
import {
  CREDENTIAL_KIND_OAUTH,
  CREDENTIAL_KIND_PAT,
  createCredentialStore,
  isValidPatFormat,
} from "./mars-credentials.mjs";
import { revokeToken, signInWithDigitalOcean } from "./mars-oauth.mjs";

export const MARS_TUNNEL_IPC = {
  openTunnel: "mars:openTunnel",
  closeTunnel: "mars:closeTunnel",
  getTunnel: "mars:getTunnel",
  getAuthState: "mars:getAuthState",
  signInWithOAuth: "mars:signInWithOAuth",
  savePat: "mars:savePat",
  setActiveConnection: "mars:setActiveConnection",
  signOut: "mars:signOut",
  listSessions: "mars:listSessions",
  listAgentConfigs: "mars:listAgentConfigs",
  listConfigSessions: "mars:listConfigSessions",
  createSession: "mars:createSession",
  pauseSession: "mars:pauseSession",
  resumeSession: "mars:resumeSession",
};

/**
 * Provisioning a brand-new sandbox is slower than waking a paused one, so the
 * create path gets a longer budget than ensureSessionAwake's wake default.
 */
const CREATE_SESSION_READY_TIMEOUT_MS = 300_000;

/**
 * Shipped MARS defaults from `config/defaults.json`. The packaged app copies
 * `config/` beside `scripts/`, so the same relative path works in dev and in
 * the installed bundle. A missing file only costs the OAuth button.
 *
 * @returns {{ apiBaseUrl?: string, oauthClientId?: string }}
 */
export function readMarsDefaults() {
  try {
    const defaultsPath = fileURLToPath(
      new URL("../config/defaults.json", import.meta.url),
    );
    return JSON.parse(readFileSync(defaultsPath, "utf-8")).mars ?? {};
  } catch {
    return {};
  }
}

/**
 * Env wins over shipped defaults so staging or a private OAuth app can be
 * used without rebuilding.
 *
 * @param {Record<string, string | undefined>} [env]
 * @param {{ apiBaseUrl?: string, oauthClientId?: string }} [defaults]
 */
export function readMarsConfig(
  env = process.env,
  defaults = readMarsDefaults(),
) {
  return {
    apiBaseUrl:
      env.MARS_API_BASE_URL || defaults.apiBaseUrl || DEFAULT_MARS_API_BASE_URL,
    oauthClientId: env.DO_OAUTH_CLIENT_ID || defaults.oauthClientId || null,
  };
}

/**
 * @param {object} [options]
 * @param {ReturnType<typeof createTunnelRegistry>} [options.registry] Override for tests
 * @param {ReturnType<typeof createCredentialStore>} [options.credentials] Override for tests
 * @param {ReturnType<typeof createMarsApiClient>} [options.api] Override for tests
 * @param {string} [options.userDataPath] Where the credential store lives
 * @param {import("electron").SafeStorage | null} [options.safeStorage]
 * @param {(url: string) => Promise<void>} [options.openExternal] Opens the OAuth page
 * @param {typeof ensureSessionAwake} [options.ensureAwake] Override for tests
 * @param {Record<string, string | undefined>} [options.env]
 */
export function createMarsTunnelBridge({
  registry = createTunnelRegistry(),
  credentials,
  api,
  userDataPath,
  safeStorage = null,
  openExternal = async () => {},
  ensureAwake = ensureSessionAwake,
  env = process.env,
} = {}) {
  const config = readMarsConfig(env);
  const store =
    credentials ?? createCredentialStore({ userDataPath, safeStorage });
  const client =
    api ??
    createMarsApiClient({
      baseUrl: config.apiBaseUrl,
      deviceId: store.deviceId,
      getToken: async () => store.getActiveToken(),
    });

  /** Configs are immutable, so a config's agent never needs re-reading. */
  const agentByConfigId = new Map();

  /**
   * The list projection has no manifest, so each config's agent is read from
   * its full record. A failed read leaves `agent` null (unknown) and is
   * retried on the next list rather than cached.
   */
  async function withManifestAgent(config) {
    if (!agentByConfigId.has(config.id)) {
      try {
        const full = await client.getAgentConfig(config.id);
        agentByConfigId.set(config.id, readManifestAgent(full?.manifest));
      } catch {
        return { ...config, agent: null };
      }
    }
    return { ...config, agent: agentByConfigId.get(config.id) };
  }

  function authState() {
    return {
      connections: store.list(),
      active: store.getActive(),
      isPersistent: store.isPersistent,
      canUseOAuth: Boolean(config.oauthClientId),
    };
  }

  function requireToken() {
    const token = store.getActiveToken();
    if (!token) {
      throw new MarsApiError("Sign in to DigitalOcean first.", { status: 401 });
    }
    return token;
  }

  /**
   * A well-formed token can still be refused by the MARS feature flippers, so
   * a new credential only sticks once a real API call has accepted it.
   */
  async function saveVerified(connection) {
    store.save(connection);
    try {
      await client.verifyAccess();
    } catch (error) {
      store.remove(store.getActive().id);
      throw error;
    }
    return authState();
  }

  return {
    registry,

    getAuthState: authState,

    async signInWithOAuth() {
      const grant = await signInWithDigitalOcean({
        clientId: config.oauthClientId,
        openExternal,
      });
      return saveVerified({
        kind: CREDENTIAL_KIND_OAUTH,
        token: grant.accessToken,
        expiresAt: grant.expiresAt,
        label: "DigitalOcean",
      });
    },

    async savePat({ token, label } = {}) {
      const trimmed = token?.trim() ?? "";
      if (!isValidPatFormat(trimmed)) {
        throw new Error(
          "That does not look like a DigitalOcean personal access token.",
        );
      }
      return saveVerified({
        kind: CREDENTIAL_KIND_PAT,
        token: trimmed,
        label: label?.trim() || "DigitalOcean token",
      });
    },

    setActiveConnection(id) {
      store.setActive(id);
      return authState();
    },

    async signOut(id) {
      const target = id ?? store.getActive()?.id;
      if (!target) return authState();
      const isActive = store.getActive()?.id === target;
      const kind = store.list().find((c) => c.id === target)?.kind;
      if (isActive && kind === CREDENTIAL_KIND_OAUTH) {
        const token = store.getActiveToken();
        if (token) await revokeToken(token);
      }
      store.remove(target);
      return authState();
    },

    listSessions: (options) => client.listSessions(options ?? {}),
    async listAgentConfigs(options) {
      const page = await client.listAgentConfigs(options ?? {});
      return {
        ...page,
        configs: await Promise.all(page.configs.map(withManifestAgent)),
      };
    },
    listConfigSessions: (configId, options) =>
      client.listConfigSessions(configId, options ?? {}),
    pauseSession: (sessionId) => client.pauseSession(sessionId),
    resumeSession: (sessionId) => client.resumeSession(sessionId),

    /**
     * Launch a session from an agent config and resolve only once it is
     * READY, so the caller can tunnel to it directly. A created session
     * starts PROVISIONING, which ensureSessionAwake already polls through.
     */
    async createSession(configId, name) {
      const session = await client.createSessionFromConfig(configId, name);
      const sessionId = session?.session_id;
      if (!sessionId) {
        throw new MarsApiError("DigitalOcean returned no session.");
      }
      return ensureAwake({
        apiUrl: config.apiBaseUrl,
        sessionId,
        accessToken: requireToken(),
        timeoutMs: CREATE_SESSION_READY_TIMEOUT_MS,
      });
    },

    /**
     * Open (or reuse) the tunnel to a session's agent-server. The guest port
     * is fixed here rather than accepted over IPC so the renderer cannot dial
     * arbitrary ports inside the sandbox.
     */
    openTunnel({ sessionId, localPort } = {}) {
      return registry.attach({
        sessionId,
        remotePort: AGENT_SERVER_GUEST_PORT,
        accessToken: requireToken(),
        apiUrl: config.apiBaseUrl,
        localPort,
      });
    },

    closeTunnel: (sessionId) => registry.detach(sessionId),
    getTunnel: (sessionId) => registry.get(sessionId),

    /** @param {import("electron").IpcMain} ipcMain */
    registerIpc(ipcMain) {
      const handle = (channel, fn) =>
        ipcMain.handle(channel, (_event, ...args) => fn(...args));
      handle(MARS_TUNNEL_IPC.openTunnel, (params) => this.openTunnel(params));
      handle(MARS_TUNNEL_IPC.closeTunnel, (sessionId) =>
        this.closeTunnel(sessionId),
      );
      handle(MARS_TUNNEL_IPC.getTunnel, (sessionId) =>
        this.getTunnel(sessionId),
      );
      handle(MARS_TUNNEL_IPC.getAuthState, () => this.getAuthState());
      handle(MARS_TUNNEL_IPC.signInWithOAuth, () => this.signInWithOAuth());
      handle(MARS_TUNNEL_IPC.savePat, (payload) => this.savePat(payload));
      handle(MARS_TUNNEL_IPC.setActiveConnection, (id) =>
        this.setActiveConnection(id),
      );
      handle(MARS_TUNNEL_IPC.signOut, (id) => this.signOut(id));
      handle(MARS_TUNNEL_IPC.listSessions, (options) =>
        this.listSessions(options),
      );
      handle(MARS_TUNNEL_IPC.listAgentConfigs, (options) =>
        this.listAgentConfigs(options),
      );
      handle(MARS_TUNNEL_IPC.listConfigSessions, (configId, options) =>
        this.listConfigSessions(configId, options),
      );
      handle(MARS_TUNNEL_IPC.createSession, (configId, name) =>
        this.createSession(configId, name),
      );
      handle(MARS_TUNNEL_IPC.pauseSession, (sessionId) =>
        this.pauseSession(sessionId),
      );
      handle(MARS_TUNNEL_IPC.resumeSession, (sessionId) =>
        this.resumeSession(sessionId),
      );
    },

    /** Tear down every open tunnel — called on app quit. */
    dispose() {
      return registry.detachAll();
    },
  };
}
