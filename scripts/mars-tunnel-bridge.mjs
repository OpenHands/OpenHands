/**
 * Main-process service behind the DigitalOcean Managed Agents screen.
 *
 * Wires the MARS REST client, the port-forward tunnel registry
 * (scripts/tunnel-registry.mjs, MARSOHS-1428/1429) and the DigitalOcean
 * credential store into Electron's IPC layer. Channel names mirror the
 * teammate fork's `window.marsBridge` surface so the two integrations stay
 * easy to reconcile.
 *
 * Connecting to a session opens a port-forward tunnel: a loopback listener
 * this process holds, relayed over harness-api's port-forward WebSocket to
 * the agent server inside the sandbox. The renderer registers the listener
 * as an ordinary local backend.
 *
 * Tokens never cross the context bridge: the renderer asks to connect by
 * session id alone, and this process sources the bearer token from the
 * stored connection for every dial. harness-api sends no CORS headers, so
 * every MARS control-plane call has to happen here anyway.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createTunnelRegistry } from "./tunnel-registry.mjs";
import { ensureSessionAwake } from "./mars-session.mjs";
import { prepareSandbox } from "./mars-sandbox-prep.mjs";
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
import {
  createSignInAttempts,
  exchangeCode,
  readCallback,
  refreshGrant,
  revokeToken,
  signInWithDigitalOcean,
} from "./mars-oauth.mjs";

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
  listSandboxSizes: "mars:listSandboxSizes",
  createOpenHandsAgent: "mars:createOpenHandsAgent",
  createSession: "mars:createSession",
  pauseSession: "mars:pauseSession",
  resumeSession: "mars:resumeSession",
  destroySession: "mars:destroySession",
  deleteAgentConfig: "mars:deleteAgentConfig",
};

/**
 * Provisioning a brand-new sandbox is slower than waking a paused one, so the
 * create path gets a longer budget than ensureSessionAwake's wake default.
 */
const CREATE_SESSION_READY_TIMEOUT_MS = 300_000;

/**
 * createSession has just waited for READY, so the openTunnel that follows
 * within this window skips one getSession round-trip. Only that first open
 * uses it: any later one — including reusing a live tunnel — asks again, so
 * a session paused in the meantime (idle, or from another client) is woken.
 */
const READY_FRESH_MS = 10_000;

/** An OAuth access token is refreshed this long before it would expire. */
const REFRESH_EARLY_MS = 60_000;

/** How long `awaitSignIn` holds a call open before answering "pending". */
const AWAIT_SIGN_IN_MS = 20_000;

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
 * @param {import("./mars-secret-store.mjs").SecretStore | null} [options.secrets]
 *   Where the Agent Canvas server keeps tokens (keychain, encrypted file, memory)
 * @param {(url: string) => Promise<void>} [options.openExternal] Opens the OAuth page
 * @param {typeof ensureSessionAwake} [options.ensureAwake] Override for tests
 * @param {typeof prepareSandbox} [options.prepare] Override for tests
 * @param {Record<string, string | undefined>} [options.env]
 */
export function createMarsTunnelBridge({
  registry: registryOverride,
  credentials,
  api,
  userDataPath,
  safeStorage = null,
  secrets = null,
  openExternal = async () => {},
  ensureAwake = ensureSessionAwake,
  prepare = prepareSandbox,
  env = process.env,
} = {}) {
  const config = readMarsConfig(env);
  const store =
    credentials ??
    createCredentialStore({ userDataPath, safeStorage, secrets });
  /** @param {() => string | null | Promise<string | null>} getToken */
  const clientFor = (getToken) =>
    createMarsApiClient({
      baseUrl: config.apiBaseUrl,
      deviceId: store.deviceId,
      getToken: async () => getToken(),
    });

  /** One refresh per connection at a time: a refresh token works only once. */
  const refreshing = new Map();

  /** Whether a grant's access token is close enough to expiry to renew. */
  function refreshDue(grant) {
    if (!grant?.refreshToken || !grant.accessExpiresAt) return false;
    const expiresAt = Date.parse(grant.accessExpiresAt);
    return (
      !Number.isNaN(expiresAt) && expiresAt - Date.now() <= REFRESH_EARLY_MS
    );
  }

  /**
   * A connection's bearer token, refreshed first when it is about to expire.
   * A grant DigitalOcean no longer honours is forgotten, so the UI asks for
   * sign-in; a network failure is thrown and the saved grant kept.
   *
   * @param {string | null | undefined} id
   * @returns {Promise<string | null>}
   */
  async function tokenFor(id) {
    if (!id) return null;
    const grant = store.getGrant(id);
    if (!grant) return null;
    if (!refreshDue(grant)) return grant.token;
    let pending = refreshing.get(id);
    if (!pending) {
      pending = refresh(id).finally(() => refreshing.delete(id));
      refreshing.set(id, pending);
    }
    return pending;
  }

  /**
   * Renew a connection's grant. Every Canvas process sharing the store takes
   * the same lock and re-reads the grant under it, so a refresh token is
   * spent once. A grant replaced meanwhile, by a new sign-in for the team,
   * is never overwritten or forgotten by this late answer.
   */
  function refresh(id) {
    return store.withGrantLock(id, async () => {
      const current = store.readGrant(id);
      if (!current) return null;
      if (!refreshDue(current)) return current.token;
      const clientId = current.clientId ?? config.oauthClientId;
      // The store writes only while it still holds the grant renewed here.
      const replacing = { replacing: current.refreshToken };
      let next;
      try {
        next = await refreshGrant({
          clientId,
          refreshToken: current.refreshToken,
        });
      } catch (error) {
        if (!error?.isRevoked) throw error;
        if (!store.updateGrant(id, null, replacing)) {
          return store.getGrant(id)?.token ?? null;
        }
        return null;
      }
      const renewed = store.updateGrant(
        id,
        {
          token: next.accessToken,
          // DigitalOcean always sends a new one; without it, a null here
          // would never be refreshed again.
          refreshToken: next.refreshToken ?? current.refreshToken,
          accessExpiresAt: next.accessExpiresAt,
          clientId,
        },
        replacing,
      );
      if (!renewed) {
        // The newer grant stays; the one renewed here is ended.
        void revokeToken(next.accessToken);
        return store.getGrant(id)?.token ?? null;
      }
      return next.accessToken;
    });
  }

  /**
   * End an OAuth grant nothing here holds any more. DigitalOcean revokes
   * access tokens only, so an expired one is refreshed first, as on
   * sign-out. A refresh refused meanwhile leaves nothing to end.
   *
   * @param {import("./mars-credentials.mjs").Grant} grant
   */
  async function endGrant(grant) {
    let token = grant.token;
    if (refreshDue(grant)) {
      try {
        const next = await refreshGrant({
          clientId: grant.clientId ?? config.oauthClientId,
          refreshToken: grant.refreshToken,
        });
        token = next.accessToken;
      } catch {
        return;
      }
    }
    await revokeToken(token);
  }

  const client = api ?? clientFor(() => tokenFor(store.getActive()?.id));
  const attempts = createSignInAttempts();

  /** When createSession saw each session READY; see READY_FRESH_MS. */
  const readyAt = new Map();
  async function wake(options) {
    const session = await ensureAwake(options);
    readyAt.set(options.sessionId, Date.now());
    return session;
  }

  const registry =
    registryOverride ??
    createTunnelRegistry({
      ensureAwake: async ({ sessionId, getAccessToken }) => {
        // Consumed by the first open after createSession, never refreshed.
        const at = readyAt.get(sessionId);
        readyAt.delete(sessionId);
        if (at !== undefined && Date.now() - at < READY_FRESH_MS) return;
        await ensureAwake({ api: clientFor(getAccessToken), sessionId });
      },
    });

  /**
   * Sessions whose sandbox is prepared: the fixes persist with the sandbox.
   * Only success is remembered; an unreachable guest, or one still booting
   * when a step failed, is tried again next time.
   */
  const preparedSessions = new Set();
  /**
   * Runs in flight, shared by concurrent connects. Keyed by listener too: a
   * run probing a listener that has since closed must not answer for the
   * tunnel that replaced it on another port.
   */
  const preparing = new Map();

  function prepareOnce(sessionId, localPort) {
    if (preparedSessions.has(sessionId)) return Promise.resolve("prepared");
    const runKey = `${sessionId}@${localPort}`;
    let run = preparing.get(runKey);
    if (!run) {
      run = prepare({
        baseUrl: buildTunnelHost(localPort),
        readFailure: () => registry.get(sessionId)?.upstreamFailure,
      })
        .then((result) => {
          if (result === "prepared") preparedSessions.add(sessionId);
          return result;
        })
        .finally(() => preparing.delete(runKey));
      preparing.set(runKey, run);
    }
    return run;
  }

  /** Configs are immutable, so a config's agent never needs re-reading. */
  const agentByConfigId = new Map();

  /**
   * Connect over the port-forward tunnel. `host` is what the renderer
   * registers as the backend's base URL: the loopback listener this process
   * holds for the session.
   */
  async function connectSession({ sessionId, localPort, connectionId }) {
    const getAccessToken = () => tokenFor(connectionId);
    // The guest port is fixed here rather than accepted over IPC so the
    // renderer cannot dial arbitrary ports inside the sandbox.
    let tunnel;
    try {
      tunnel = await registry.attach({
        sessionId,
        remotePort: AGENT_SERVER_GUEST_PORT,
        getAccessToken,
        apiUrl: config.apiBaseUrl,
        localPort,
        owner: connectionId,
      });
    } catch (error) {
      // A failed dial may mean the session is no longer READY; ask again.
      readyAt.delete(sessionId);
      throw error;
    }
    // Every caller gets a prepared sandbox, not just the Canvas page, and
    // learns how that went: `sandbox` is the outcome of prepareSandbox.
    let sandbox;
    if (tunnel.status === "connected" && tunnel.localPort !== undefined) {
      sandbox = await prepareOnce(sessionId, tunnel.localPort);
      // Preparing can take a minute; answer with the tunnel as it is now,
      // since it may have been closed or replaced meanwhile.
      tunnel = registry.get(sessionId) ?? {
        ...tunnel,
        status: "error",
        localPort: undefined,
        error: "The session was disconnected while its sandbox was prepared.",
      };
    }
    return {
      ...tunnel,
      sandbox,
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
      storage: store.storage,
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
   *
   * Once accepted, the account names the team: from the OAuth reply when
   * there is one, else from the account call. A team that already has a
   * connection keeps it (see `setTeam`). A token that may not read the
   * account keeps its label instead.
   */
  async function saveVerified(
    connection,
    account = null,
    isCancelled = () => false,
  ) {
    const previousId = store.getActive()?.id ?? null;
    const saved = store.save(connection);
    // Ask about this token, not whichever connection is active when
    // DigitalOcean answers: another tab or agent may switch teams meanwhile.
    const pinned = api ?? clientFor(() => connection.token);
    let team;
    try {
      await pinned.verifyAccess();
      team = account ?? (await pinned.getAccount?.().catch(() => null));
      // The last point a cancelled sign-in can still be undone.
      if (isCancelled()) {
        throw new MarsApiError("Sign-in was cancelled.", { status: 409 });
      }
    } catch (error) {
      const stillActive = store.getActive()?.id === saved.id;
      if (!saved.reused) store.remove(saved.id);
      if (previousId && stillActive) store.setActive(previousId);
      throw error;
    }
    // The credential is kept from here on, so naming its team must not
    // fail the sign-in; a team already held then just keeps both entries.
    if (team) {
      try {
        store.setTeam(saved.id, team, {
          onReplacedGrant: (replaced) => void endGrant(replaced),
        });
      } catch (error) {
        console.warn(
          `[mars] Could not record the DigitalOcean team: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    }
    return authState();
  }

  /** @param {Awaited<ReturnType<typeof exchangeCode>>} grant */
  function saveOAuthGrant(grant, clientId, isCancelled) {
    return saveVerified(
      {
        kind: CREDENTIAL_KIND_OAUTH,
        token: grant.accessToken,
        refreshToken: grant.refreshToken,
        accessExpiresAt: grant.accessExpiresAt,
        clientId,
        label: "DigitalOcean",
        teamName: grant.account?.name ?? null,
      },
      grant.account,
      isCancelled,
    );
  }

  function requireOAuthClient() {
    if (!config.oauthClientId) {
      throw new MarsApiError(
        "Sign-in with DigitalOcean is not set up on this Canvas. Use a personal access token instead.",
        { status: 400 },
      );
    }
    return config.oauthClientId;
  }

  return {
    registry,

    getAuthState: authState,

    /** The desktop app's sign-in: system browser, loopback return. */
    async signInWithOAuth() {
      const clientId = requireOAuthClient();
      const grant = await signInWithDigitalOcean({ clientId, openExternal });
      try {
        return await saveOAuthGrant(grant, clientId);
      } catch (failure) {
        // Nothing keeps a grant that failed to save: end it.
        void revokeToken(grant.accessToken);
        throw failure;
      }
    },

    /**
     * Start a browser sign-in. The page opens `authorizeUrl`; DigitalOcean
     * sends the browser back to `redirectUri`, which the Canvas server
     * passes to `completeSignIn`. The PKCE proof stays here.
     *
     * @param {{ redirectUri: string }} options
     */
    beginSignIn({ redirectUri } = {}) {
      const clientId = requireOAuthClient();
      if (!redirectUri) {
        throw new MarsApiError("A sign-in needs a return address.", {
          status: 400,
        });
      }
      return attempts.begin({ clientId, redirectUri });
    },

    /**
     * How a browser sign-in stands, waiting a little for it to finish. The
     * page calls again while it reads "pending".
     *
     * @param {string} attemptId
     */
    async awaitSignIn(attemptId) {
      const result = await attempts.wait(attemptId, AWAIT_SIGN_IN_MS);
      return result.status === "done"
        ? { ...result, authState: authState() }
        : result;
    },

    /**
     * Finish a browser sign-in from the address DigitalOcean sent the
     * browser to: by the Canvas server's return route, or pasted by the user
     * when that address could not load. Works once, for an open attempt.
     *
     * @param {string} returned The return address, or just its query string
     * @param {string} [attemptId] When given, the return must belong to it
     */
    async completeSignIn(returned, attemptId) {
      const { code, state, error } = readCallback(returned);
      // Checked before the attempt is taken, so the right address can
      // still follow a wrong one.
      if (!code && !error) {
        throw new MarsApiError("That address has no sign-in code in it.", {
          status: 400,
        });
      }
      const attempt = attempts.claim(state, attemptId);
      if (!attempt) {
        throw new MarsApiError(
          attemptId && attempts.isOpen(attemptId)
            ? "That address isn't from this sign-in. Paste the address DigitalOcean opened after you approved."
            : "This sign-in is no longer open. Start again.",
          { status: 400 },
        );
      }
      // Cancel stays possible until the grant is committed.
      const cancelled = () => attempt.cancelled === true;
      let grant = null;
      try {
        if (error) throw new MarsApiError(error, { status: 400 });
        grant = await exchangeCode({
          clientId: attempt.clientId,
          code,
          verifier: attempt.verifier,
          redirectUri: attempt.redirectUri,
        });
        if (cancelled()) {
          throw new MarsApiError("Sign-in was cancelled.", { status: 409 });
        }
        const signedIn = await saveOAuthGrant(
          grant,
          attempt.clientId,
          cancelled,
        );
        attempts.succeed(attempt.id);
        return signedIn;
      } catch (failure) {
        // Nothing keeps a grant whose sign-in failed or was cancelled after
        // the exchange (a team without Managed Agents, say): end it.
        if (grant) void revokeToken(grant.accessToken);
        attempts.fail(
          attempt.id,
          failure instanceof Error ? failure.message : String(failure),
        );
        throw failure;
      }
    },

    /** @param {string} attemptId */
    cancelSignIn(attemptId) {
      attempts.cancel(attemptId);
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
      await registry.detachOwnedBy(target);
      readyAt.clear();
      const kind = store.list().find((c) => c.id === target)?.kind;
      if (kind === CREDENTIAL_KIND_OAUTH) {
        // DigitalOcean revokes access tokens only, so revoke a current one:
        // an hour-old token is refreshed first rather than left unrevoked.
        const token = await tokenFor(target).catch(() => null);
        if (token && !(await revokeToken(token))) {
          console.warn(
            "[mars] DigitalOcean did not confirm the sign-out; the sign-in is removed here anyway.",
          );
        }
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
    async listSandboxSizes() {
      requireActiveConnectionId();
      return client.listSandboxSizes();
    },
    pauseSession(sessionId) {
      readyAt.delete(sessionId);
      return client.pauseSession(sessionId);
    },
    resumeSession: (sessionId) => client.resumeSession(sessionId),

    /**
     * Destroy first, then drop the live connection: harness-api can refuse
     * (409 while a checkpoint/fork/rollback holds the session, 423 locked),
     * and then the session is still running and the user must keep their
     * connection to it. Only a session that is really gone is detached.
     */
    async destroySession(sessionId) {
      await client.destroySession(sessionId);
      readyAt.delete(sessionId);
      await registry.detach(sessionId);
    },
    async deleteAgentConfig(configId) {
      await client.deleteAgentConfig(configId);
      agentByConfigId.delete(configId);
    },

    /**
     * Create an OpenHands Agent Config. The manifest is built here rather
     * than accepted over IPC so the renderer can only ever create OpenHands
     * agents; it contributes just the name and an optional sandbox size slug
     * (validated by the manifest builder).
     */
    async createOpenHandsAgent({ name, size } = {}) {
      const trimmedName = name?.trim() ?? "";
      if (!trimmedName) {
        throw new MarsApiError("Give the agent a name.", { status: 400 });
      }
      const created = await client.createAgentConfig(
        trimmedName,
        buildOpenHandsManifest({ size: size || undefined }),
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
      return wake({
        api: client,
        sessionId,
        timeoutMs: CREATE_SESSION_READY_TIMEOUT_MS,
      });
    },

    /**
     * Connect to a session's agent-server over a port-forward tunnel. The
     * connection is bound to the active credential: tunnel dials read that
     * credential's current token and are cut off when it signs out.
     */
    openTunnel({ sessionId, localPort } = {}) {
      const connectionId = requireActiveConnectionId();
      return connectSession({ sessionId, localPort, connectionId });
    },

    closeTunnel(sessionId) {
      readyAt.delete(sessionId);
      return registry.detach(sessionId);
    },
    getTunnel(sessionId) {
      return registry.get(sessionId);
    },

    /** @param {import("electron").IpcMain} ipcMain */
    registerIpc(ipcMain) {
      for (const [method, channel] of Object.entries(MARS_TUNNEL_IPC)) {
        ipcMain.handle(channel, (_event, ...args) => this[method](...args));
      }
    },

    /** Tear down every tunnel — on quit. */
    dispose() {
      return registry.detachAll();
    },
  };
}
