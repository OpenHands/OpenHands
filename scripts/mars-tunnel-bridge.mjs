/**
 * Main-process service behind the DigitalOcean Managed Agents screen.
 *
 * Wires the MARS REST client, public ingress (scripts/mars-ingress.mjs), the
 * port-forward tunnel registry (scripts/tunnel-registry.mjs, MARSOHS-1428/
 * 1429) and the DigitalOcean credential store into Electron's IPC layer.
 * Channel names mirror the teammate fork's `window.marsBridge` surface so the
 * two integrations stay easy to reconcile.
 *
 * Connecting to a session prefers its public ingress URL: the renderer then
 * talks to the Agent Server directly (REST and WebSocket) with no local
 * listener in between. Sessions that cannot have one (harness-api 501) fall
 * back to the tunnel, so older sandboxes keep working.
 *
 * Tokens never cross the context bridge: the renderer asks to connect by
 * session id alone, this process sources the bearer token from the stored
 * connection, and for ingress it injects that token on the renderer's own
 * requests to the public host (`registerRequestAuth`). harness-api sends no
 * CORS headers, so every MARS control-plane call has to happen here anyway.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createTunnelRegistry } from "./tunnel-registry.mjs";
import { ensureSessionAwake } from "./mars-session.mjs";
import {
  MarsIngressUnsupportedError,
  resolveIngressURL,
} from "./mars-ingress.mjs";
import {
  AGENT_SERVER_GUEST_PORT,
  DEFAULT_MARS_API_BASE_URL,
  MarsApiError,
  OPENHANDS_AGENT,
  buildOpenHandsManifest,
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
  createOpenHandsAgent: "mars:createOpenHandsAgent",
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
 * Only the renderer's requests to a connected session's public host get the
 * bearer injected. Scoped to TLS schemes: an ingress URL is always https, and
 * its WebSocket upgrade wss.
 */
const INGRESS_REQUEST_FILTER = { urls: ["https://*/*", "wss://*/*"] };

export function buildTunnelHost(localPort) {
  return `http://127.0.0.1:${localPort}`;
}

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
 * @param {typeof resolveIngressURL} [options.resolveIngress] Override for tests
 * @param {Record<string, string | undefined>} [options.env]
 */
export function createMarsTunnelBridge({
  registry: registryOverride,
  credentials,
  api,
  userDataPath,
  safeStorage = null,
  openExternal = async () => {},
  ensureAwake = ensureSessionAwake,
  resolveIngress = resolveIngressURL,
  env = process.env,
} = {}) {
  const config = readMarsConfig(env);
  const store =
    credentials ?? createCredentialStore({ userDataPath, safeStorage });
  /** @param {() => string | null} getToken */
  const clientFor = (getToken) =>
    createMarsApiClient({
      baseUrl: config.apiBaseUrl,
      deviceId: store.deviceId,
      getToken: async () => getToken(),
    });
  const client = api ?? clientFor(() => store.getActiveToken());
  const registry =
    registryOverride ??
    createTunnelRegistry({
      ensureAwake: ({ sessionId, getAccessToken }) =>
        ensureAwake({ api: clientFor(getAccessToken), sessionId }),
    });

  /** Configs are immutable, so a config's agent never needs re-reading. */
  const agentByConfigId = new Map();

  /**
   * Sessions connected over public ingress, by session id. `host` is the
   * URL's authority (`hostname[:port]`), matched against the renderer's
   * outgoing requests to decide which get the bearer. `connectionId` names
   * the credential the session was connected with, read live on every
   * request so sign-out cuts it off exactly as it does for a tunnel.
   *
   * @type {Map<string, { host: string, connectionId: string, status: object }>}
   */
  const ingressSessions = new Map();

  function dropIngressOwnedBy(connectionId) {
    for (const [sessionId, entry] of ingressSessions) {
      if (entry.connectionId === connectionId)
        ingressSessions.delete(sessionId);
    }
  }

  /**
   * Connect over public ingress, or over the port-forward tunnel when the
   * session cannot have a URL. Both resolve to the same status shape; `host`
   * is what the renderer registers as the backend's base URL.
   */
  async function connectSession({ sessionId, localPort, connectionId }) {
    const getAccessToken = () => store.getToken(connectionId);
    try {
      const ingress = await resolveIngress({
        api: clientFor(getAccessToken),
        sessionId,
        ensureAwake,
      });
      const status = {
        sessionId,
        status: "connected",
        transport: "ingress",
        host: ingress.url,
        ingressUrlId: ingress.ingressUrlId,
        remotePort: ingress.port,
        localPort: undefined,
        error: undefined,
        upstreamFailure: null,
      };
      ingressSessions.set(sessionId, {
        host: new URL(ingress.url).host,
        connectionId,
        status,
      });
      return status;
    } catch (error) {
      if (!(error instanceof MarsIngressUnsupportedError)) throw error;
    }
    // The guest port is fixed here rather than accepted over IPC so the
    // renderer cannot dial arbitrary ports inside the sandbox.
    const tunnel = await registry.attach({
      sessionId,
      remotePort: AGENT_SERVER_GUEST_PORT,
      getAccessToken,
      apiUrl: config.apiBaseUrl,
      localPort,
      owner: connectionId,
    });
    return {
      ...tunnel,
      transport: "tunnel",
      host:
        tunnel.localPort === undefined
          ? undefined
          : buildTunnelHost(tunnel.localPort),
    };
  }

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

  function requireActiveConnectionId() {
    const id = store.getActive()?.id;
    if (!id || !store.getToken(id)) {
      throw new MarsApiError("Sign in to DigitalOcean first.", { status: 401 });
    }
    return id;
  }

  /**
   * A well-formed token can still be refused by the MARS feature flippers, so
   * a new credential only sticks once a real API call has accepted it. A
   * rejected one leaves the previously active connection active.
   */
  async function saveVerified(connection) {
    const previousId = store.getActive()?.id ?? null;
    const saved = store.save(connection);
    try {
      await client.verifyAccess();
    } catch (error) {
      store.remove(saved.id);
      if (previousId) store.setActive(previousId);
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

    /**
     * Open tunnels are left alone: each keeps dialing with the connection
     * that opened it, since its session belongs to that connection's team.
     */
    setActiveConnection(id) {
      store.setActive(id);
      return authState();
    },

    async signOut(id) {
      const target = id ?? store.getActive()?.id;
      if (!target) return authState();
      dropIngressOwnedBy(target);
      await registry.detachOwnedBy(target);
      const kind = store.list().find((c) => c.id === target)?.kind;
      if (kind === CREDENTIAL_KIND_OAUTH) {
        const token = store.getToken(target);
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
     * Create an OpenHands Agent Config. The manifest is built here rather
     * than accepted over IPC so the renderer can only ever create OpenHands
     * agents; it contributes just the name and an optional LLM key.
     */
    async createOpenHandsAgent({ name, llmApiKey } = {}) {
      const trimmedName = name?.trim() ?? "";
      if (!trimmedName) {
        throw new MarsApiError("Give the agent a name.", { status: 400 });
      }
      const created = await client.createAgentConfig(
        trimmedName,
        buildOpenHandsManifest({ llmApiKey: llmApiKey?.trim() || undefined }),
      );
      if (!created?.id) {
        throw new MarsApiError("DigitalOcean returned no agent.");
      }
      agentByConfigId.set(created.id, OPENHANDS_AGENT);
      return {
        id: created.id,
        name: created.name ?? trimmedName,
        created_by: created.created_by ?? null,
        updated_at: created.updated_at ?? null,
        agent: OPENHANDS_AGENT,
      };
    },

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
        api: client,
        sessionId,
        timeoutMs: CREATE_SESSION_READY_TIMEOUT_MS,
      });
    },

    /**
     * Connect to a session's agent-server: over its public ingress URL when
     * the session has one (re-resolved every time, since the hostname is
     * revoked on pause and lock), otherwise over a port-forward tunnel. The
     * connection is bound to the active credential: ingress requests and
     * tunnel dials read that credential's current token, and both are cut
     * off when it signs out. Kept under the historical `openTunnel` name so
     * persisted backends and the renderer's restore path need no migration.
     */
    openTunnel({ sessionId, localPort } = {}) {
      const connectionId = requireActiveConnectionId();
      return connectSession({ sessionId, localPort, connectionId });
    },

    closeTunnel(sessionId) {
      ingressSessions.delete(sessionId);
      return registry.detach(sessionId);
    },
    getTunnel(sessionId) {
      return ingressSessions.get(sessionId)?.status ?? registry.get(sessionId);
    },

    /**
     * `Authorization` value for a renderer request to a connected session's
     * public host, or null when the URL is not one of ours or that
     * credential is gone. Matched on the URL authority so the WebSocket
     * upgrade (`wss://`) and REST (`https://`) to the same host both qualify.
     */
    ingressAuthorizationHeader(url) {
      let host;
      try {
        ({ host } = new URL(url));
      } catch {
        return null;
      }
      for (const entry of ingressSessions.values()) {
        if (entry.host !== host) continue;
        const token = store.getToken(entry.connectionId);
        return token ? `Bearer ${token}` : null;
      }
      return null;
    },

    /**
     * Inject the bearer on the renderer's own requests to connected ingress
     * hosts. Runs in the main process for every matching request, including
     * WebSocket handshakes, which is what lets the renderer open
     * `new WebSocket(url)` against a PAT-authenticated host without ever
     * seeing the token. Requests to any other host pass through untouched.
     *
     * @param {import("electron").Session} electronSession
     */
    registerRequestAuth(electronSession) {
      electronSession.webRequest.onBeforeSendHeaders(
        INGRESS_REQUEST_FILTER,
        (details, callback) => {
          const authorization = this.ingressAuthorizationHeader(details.url);
          callback({
            requestHeaders: authorization
              ? { ...details.requestHeaders, Authorization: authorization }
              : details.requestHeaders,
          });
        },
      );
    },

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
      handle(MARS_TUNNEL_IPC.createOpenHandsAgent, (payload) =>
        this.createOpenHandsAgent(payload),
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

    /** Forget every ingress connection and tear down every tunnel — on quit. */
    dispose() {
      ingressSessions.clear();
      return registry.detachAll();
    },
  };
}
