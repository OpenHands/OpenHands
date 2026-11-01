/**
 * Store for DigitalOcean credentials used by the MARS backend.
 *
 * Holds *multiple* named connections rather than a single token. An OAuth
 * access token is pinned to one team (`team_uuid` in the grant) and MARS
 * sessions are team-scoped, so working across teams means holding one
 * credential per team and switching between them.
 *
 * Secrets live in one of two places. The Agent Canvas server hands in a
 * `SecretStore` (scripts/mars-secret-store.mjs: the OS keychain, an encrypted
 * file, or memory), and the metadata file holds no token at all. The desktop
 * app still passes Electron's `safeStorage`, and then only the token is
 * encrypted, inline in the metadata file.
 *
 * The surrounding metadata stays readable so the UI can label connections
 * without touching a secret. Nothing here is ever handed to the renderer
 * except through `toMetadata`, which omits the token.
 */

import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { hostname, platform } from "node:os";
import path from "node:path";

import { withLockFile, withLockFileSync } from "./mars-lock-file.mjs";

const STORE_FILENAME = "mars-credentials.json";
const STORE_VERSION = 1;

export const CREDENTIAL_KIND_OAUTH = "oauth";
export const CREDENTIAL_KIND_PAT = "pat";

const LEGACY_PAT_LENGTH = 64;
const V1_PAT_LENGTH = 71;
const V1_PAT_PREFIX = "do";

/** Account name a connection's secret is stored under in a `SecretStore`. */
export function secretAccountFor(connectionId) {
  return `digitalocean/${connectionId}`;
}

/** Version of the record kept for each connection's secret. */
const SECRET_VERSION = 1;

/**
 * @typedef {object} Grant What a connection signs in with.
 * @property {string} token The bearer token: a personal access token, or an
 *   OAuth access token.
 * @property {string | null} [refreshToken] OAuth only; works exactly once.
 * @property {string | null} [accessExpiresAt] OAuth only; when `token` stops
 *   working and must be refreshed.
 * @property {string | null} [clientId] The OAuth client it was issued to,
 *   which a refresh must name.
 */

/** @param {string} kind @param {Grant} grant */
function secretFor(kind, grant) {
  return JSON.stringify({
    v: SECRET_VERSION,
    kind,
    token: grant.token,
    ...(grant.refreshToken ? { refreshToken: grant.refreshToken } : {}),
    ...(grant.accessExpiresAt
      ? { accessExpiresAt: grant.accessExpiresAt }
      : {}),
    ...(grant.clientId ? { clientId: grant.clientId } : {}),
  });
}

/**
 * A stored secret as a grant; null for anything this version cannot read.
 * A bare token is what the desktop app stored before records existed.
 *
 * @returns {Grant | null}
 */
function grantFromSecret(secret) {
  if (!secret) return null;
  let record;
  try {
    record = JSON.parse(secret);
  } catch {
    return { token: secret };
  }
  if (record?.v !== SECRET_VERSION || typeof record.token !== "string") {
    return null;
  }
  return {
    token: record.token,
    refreshToken: record.refreshToken ?? null,
    accessExpiresAt: record.accessExpiresAt ?? null,
    clientId: record.clientId ?? null,
  };
}

/**
 * Shape check matching the one `doctl` performs before storing a token: 64
 * characters for legacy tokens, or 71 beginning with `do` for v1 tokens.
 *
 * This is a fast typo guard only. It cannot tell whether the token is valid or
 * whether the account has MARS access, so callers must still verify it against
 * the API before treating the connection as usable.
 */
export function isValidPatFormat(token) {
  const trimmed = token?.trim() ?? "";
  if (trimmed.length === LEGACY_PAT_LENGTH) return true;
  return trimmed.length === V1_PAT_LENGTH && trimmed.startsWith(V1_PAT_PREFIX);
}

/**
 * A short hash that recognises the same token entered twice (or `MARS_TOKEN`
 * on every start) without storing anything that could be used as it.
 */
function fingerprintOf(token) {
  return createHash("sha256").update(token).digest("hex").slice(0, 16);
}

/**
 * Held across one refresh: a token request times out after 15 s, so a lock
 * this old was abandoned.
 */
const GRANT_LOCK = {
  staleMs: 120_000,
  waitMs: 30_000,
  busyMessage:
    "Another Canvas process is still renewing this DigitalOcean sign-in. Try again.",
};

/**
 * Held while the connection list is read and written back, which may include
 * saving a secret in the keychain (its tool times out after 10 s).
 */
const STORE_LOCK = {
  staleMs: 60_000,
  waitMs: 20_000,
  busyMessage:
    "Another Canvas process is still saving a DigitalOcean sign-in. Try again.",
};

function emptyState() {
  return {
    version: STORE_VERSION,
    deviceId: randomUUID(),
    activeConnectionId: null,
    connections: [],
  };
}

function isExpired(connection) {
  if (!connection?.expiresAt) return false;
  return Date.parse(connection.expiresAt) <= Date.now();
}

/**
 * @param {object} options
 * @param {string | null} [options.userDataPath] Where the metadata file lives;
 *   null keeps everything in memory for this process.
 * @param {Pick<import("electron").SafeStorage, "isEncryptionAvailable" | "encryptString" | "decryptString"> | null} [options.safeStorage]
 *   The desktop app's encryption; tokens are then stored inline, encrypted.
 * @param {import("./mars-secret-store.mjs").SecretStore | null} [options.secrets]
 *   Where the server keeps tokens; the metadata file then holds none.
 * @param {(message: string) => void} [options.log]
 */
export function createCredentialStore({
  userDataPath = null,
  safeStorage = null,
  secrets = null,
  log = (message) => console.warn(`[mars-credentials] ${message}`),
} = {}) {
  const filePath = userDataPath
    ? path.join(userDataPath, STORE_FILENAME)
    : null;

  // When the OS keychain is unavailable (common on bare Linux desktops),
  // safeStorage credentials live only for this process rather than being
  // written to disk in the clear.
  const canEncrypt = Boolean(
    !secrets && filePath && safeStorage?.isEncryptionAvailable?.(),
  );
  /**
   * Plaintext grant per connection id. The only copy when nothing persists
   * it; otherwise a cache, so the keychain is not hit on every dial. Null
   * records a secret that could not be read.
   *
   * @type {Map<string, Grant | null>}
   */
  const memoryGrants = new Map();
  /** Connections whose grant this process could not save: memory holds it. */
  const unsaved = new Set();

  let state = load();

  /**
   * The connection list as the file holds it now; null when there is no
   * usable file. Other Canvas processes sharing the store write it too.
   */
  function readState() {
    if (!filePath) return null;
    try {
      const parsed = JSON.parse(readFileSync(filePath, "utf8"));
      if (parsed?.version !== STORE_VERSION) return null;
      return {
        ...emptyState(),
        ...parsed,
        connections: Array.isArray(parsed.connections)
          ? parsed.connections
          : [],
      };
    } catch {
      return null;
    }
  }

  /** Read a connection's secret into the cache. */
  function readSecret(id) {
    let grant = null;
    try {
      grant = grantFromSecret(secrets.read(secretAccountFor(id)));
    } catch (error) {
      // A locked keychain: the connection stays listed, needing sign-in.
      log(error instanceof Error ? error.message : String(error));
    }
    memoryGrants.set(id, grant);
    return grant;
  }

  function load() {
    memoryGrants.clear();
    unsaved.clear();
    // Missing or corrupt store — start clean rather than blocking sign-in.
    const loaded = readState() ?? emptyState();
    if (secrets) {
      for (const connection of loaded.connections) readSecret(connection.id);
    }
    return loaded;
  }

  /**
   * Change the store as the file holds it now, under the store's lock, and
   * write it back, so a change another Canvas process made meanwhile is
   * kept. `change` returns false when it changed nothing in the file.
   *
   * @param {() => boolean | void} change
   */
  function mutate(change) {
    if (!filePath) {
      change();
      return;
    }
    withLockFileSync(
      `${filePath}.lock`,
      () => {
        state = readState() ?? state;
        if (change() !== false) persist();
      },
      STORE_LOCK,
    );
  }

  function persist() {
    mkdirSync(path.dirname(filePath), { recursive: true });
    const serializable = {
      ...state,
      connections: state.connections.map((connection) => ({
        ...connection,
        // Only safeStorage keeps a token here, and only an encrypted one.
        token: canEncrypt ? connection.token : null,
      })),
    };
    // Write the whole file, then replace the old one in one step.
    const temporary = `${filePath}.${process.pid}.tmp`;
    writeFileSync(temporary, JSON.stringify(serializable, null, 2), {
      mode: 0o600,
    });
    renameSync(temporary, filePath);
  }

  function encrypt(secret) {
    if (!canEncrypt) return null;
    return safeStorage.encryptString(secret).toString("base64");
  }

  /** @returns {Grant | null} */
  function decrypt(connection) {
    const cached = memoryGrants.get(connection.id);
    if (cached !== undefined) return cached;
    // A connection another Canvas process added since this one started.
    if (secrets) return readSecret(connection.id);
    if (!canEncrypt || !connection.token) return null;
    try {
      const grant = grantFromSecret(
        safeStorage.decryptString(Buffer.from(connection.token, "base64")),
      );
      if (grant) memoryGrants.set(connection.id, grant);
      return grant;
    } catch {
      // Keychain rotated or the profile moved between machines.
      return null;
    }
  }

  /**
   * Keep a grant where this store keeps secrets; false when it was refused.
   *
   * @param {object} connection
   * @param {Grant} grant
   */
  function storeGrant(connection, grant) {
    memoryGrants.set(connection.id, grant);
    const secret = secretFor(connection.kind, grant);
    if (!secrets) {
      connection.token = encrypt(secret);
      return true;
    }
    try {
      secrets.write(secretAccountFor(connection.id), secret);
      unsaved.delete(connection.id);
      return true;
    } catch (error) {
      // Usable for this run; the UI says it was not saved.
      log(error instanceof Error ? error.message : String(error));
      unsaved.add(connection.id);
      return false;
    }
  }

  function forgetToken(id) {
    memoryGrants.delete(id);
    unsaved.delete(id);
    if (!secrets) return;
    try {
      secrets.remove(secretAccountFor(id));
    } catch (error) {
      log(error instanceof Error ? error.message : String(error));
    }
  }

  function find(id) {
    return state.connections.find((connection) => connection.id === id) ?? null;
  }

  /**
   * The grant the store holds for `connection` now, past this process's
   * cache. One this process could not save exists only in its memory, so
   * the store's copy (none, or an older one) does not replace it.
   *
   * @param {object} connection
   * @returns {Grant | null}
   */
  function storedGrant(connection) {
    if (!secrets) return decrypt(connection);
    const kept = memoryGrants.get(connection.id);
    if (kept && unsaved.has(connection.id)) return kept;
    let grant;
    try {
      grant = grantFromSecret(secrets.read(secretAccountFor(connection.id)));
    } catch (error) {
      log(error instanceof Error ? error.message : String(error));
      return decrypt(connection);
    }
    if (grant) memoryGrants.set(connection.id, grant);
    else memoryGrants.delete(connection.id);
    return grant;
  }

  /** Public shape of a connection: everything except the secret. */
  function toMetadata(connection) {
    if (!connection) return null;
    return {
      id: connection.id,
      kind: connection.kind,
      label: connection.label,
      teamName: connection.teamName ?? null,
      email: connection.email ?? null,
      expiresAt: connection.expiresAt ?? null,
      // No readable token (a locked keychain, a lost key) means sign in again.
      isExpired: isExpired(connection) || decrypt(connection) === null,
      isSaved: connection.isSaved !== false,
    };
  }

  /**
   * Where tokens are kept, for the UI. Mirrors the SecretStore when there is
   * one; for safeStorage, the OS keychain protects the encryption key.
   */
  function storage() {
    const host = hostname();
    if (secrets) {
      return {
        kind: secrets.isPersistent && filePath ? secrets.kind : "memory",
        label: secrets.label,
        host,
      };
    }
    if (!canEncrypt) return { kind: "memory", label: "memory", host };
    const label =
      platform() === "darwin" ? "macOS Keychain" : "system keychain";
    return { kind: "keychain", label, host };
  }

  return {
    /** True when tokens survive a restart. Surfaced so the UI can warn. */
    get isPersistent() {
      return storage().kind !== "memory";
    },

    get storage() {
      return storage();
    },

    /** Stable per-install id, sent as `X-Device-UUID` the way doctl does. */
    get deviceId() {
      return state.deviceId;
    },

    list() {
      return state.connections.map(toMetadata);
    },

    getActive() {
      return toMetadata(find(state.activeConnectionId));
    },

    getActiveToken() {
      return this.getToken(state.activeConnectionId);
    },

    /** Token of one connection, or null once it is removed or expired. */
    getToken(id) {
      return this.getGrant(id)?.token ?? null;
    },

    /**
     * Everything a connection signs in with, refresh token included; null
     * once it is removed or expired. Never leaves the bridge.
     *
     * @returns {Grant | null}
     */
    getGrant(id) {
      const connection = find(id);
      if (!connection || isExpired(connection)) return null;
      return decrypt(connection);
    },

    /**
     * The grant as the store holds it now, read past this process's cache:
     * another Canvas process sharing the store may have renewed or removed
     * it. Without a SecretStore, this process is the only writer.
     *
     * @returns {Grant | null}
     */
    readGrant(id) {
      const connection = find(id);
      if (!connection || isExpired(connection)) return null;
      return storedGrant(connection);
    },

    /**
     * Run `fn` holding this connection's grant lock, which every Canvas
     * process sharing the store respects: a refresh token works only once.
     *
     * @template T
     * @param {string} id
     * @param {() => Promise<T>} fn
     * @returns {Promise<T>}
     */
    withGrantLock(id, fn) {
      if (!filePath) return fn();
      return withLockFile(
        path.join(path.dirname(filePath), `grant-${id}.lock`),
        fn,
        GRANT_LOCK,
      );
    },

    /**
     * Replace a connection's grant after a refresh. Null forgets it: the
     * connection stays listed and asks for sign-in.
     *
     * With `replacing`, only while the store still holds the grant with that
     * refresh token. That is checked in the same locked step as the write,
     * so a new sign-in another Canvas process saved meanwhile stands.
     * Returns false when the grant it would replace is gone.
     *
     * @param {string} id
     * @param {Grant | null} grant
     * @param {{ replacing?: string }} [options]
     * @returns {boolean}
     */
    updateGrant(id, grant, { replacing } = {}) {
      let updated = false;
      mutate(() => {
        const connection = find(id);
        if (!connection) return false;
        if (
          replacing !== undefined &&
          storedGrant(connection)?.refreshToken !== replacing
        ) {
          return false;
        }
        updated = true;
        const before = { token: connection.token, isSaved: connection.isSaved };
        if (grant) {
          connection.isSaved = storeGrant(connection, grant);
        } else {
          forgetToken(id);
          connection.token = null;
        }
        // With a SecretStore a renewal changes nothing in the file, unless
        // the secret could not be saved.
        return (
          connection.token !== before.token ||
          connection.isSaved !== before.isSaved
        );
      });
      return updated;
    },

    /**
     * Save a credential and make it active. Entering a token this store
     * already holds updates that connection instead of adding a second one.
     *
     * @param {Grant & { kind: string, expiresAt?: string | null, label: string, teamName?: string | null }} input
     *   `expiresAt` is when the whole credential lapses (a token that cannot be
     *   refreshed); `accessExpiresAt` only says when to refresh.
     * @returns {ReturnType<typeof toMetadata> & { reused: boolean }}
     */
    save({
      kind,
      token,
      refreshToken = null,
      accessExpiresAt = null,
      clientId = null,
      expiresAt = null,
      label,
      teamName = null,
    }) {
      const fingerprint = fingerprintOf(token);
      let connection;
      let existing;
      mutate(() => {
        existing = state.connections.find(
          (other) => other.fingerprint === fingerprint,
        );
        connection = existing ?? { id: randomUUID() };
        Object.assign(connection, {
          kind,
          label,
          teamName: teamName ?? connection.teamName ?? null,
          expiresAt,
          fingerprint,
        });
        connection.isSaved = storeGrant(connection, {
          token,
          refreshToken,
          accessExpiresAt,
          clientId,
        });
        if (!existing) state.connections = [...state.connections, connection];
        state.activeConnectionId = connection.id;
      });
      return { ...toMetadata(connection), reused: Boolean(existing) };
    },

    /**
     * Record which team a connection belongs to. A team already held by an
     * older connection keeps that connection: it takes the new token, and the
     * new entry is dropped, so anything bound to the old id carries on.
     *
     * The older connection's own OAuth grant is then held nowhere, so it is
     * handed to `onReplacedGrant` to be ended. A personal access token is
     * the person's own and is never handed over.
     *
     * @param {string} id
     * @param {{ uuid?: string | null, name?: string | null, email?: string | null }} team
     * @param {{ onReplacedGrant?: (grant: Grant) => void }} [options]
     */
    setTeam(
      id,
      { uuid = null, name = null, email = null } = {},
      { onReplacedGrant } = {},
    ) {
      let target = null;
      let replaced = null;
      mutate(() => {
        const connection = find(id);
        if (!connection) return false;
        const older = uuid
          ? state.connections.find(
              (other) => other.id !== id && other.teamUuid === uuid,
            )
          : null;
        target = older ?? connection;
        if (older) {
          const grant = decrypt(connection);
          const previous = decrypt(older);
          if (
            older.kind === CREDENTIAL_KIND_OAUTH &&
            previous &&
            previous.token !== grant?.token
          ) {
            replaced = previous;
          }
          Object.assign(older, {
            kind: connection.kind,
            label: connection.label,
            expiresAt: connection.expiresAt ?? null,
            fingerprint: connection.fingerprint,
          });
          if (grant) older.isSaved = storeGrant(older, grant);
          forgetToken(id);
          state.connections = state.connections.filter(
            (other) => other.id !== id,
          );
          // Follow the merge only if this connection was the active one; a
          // switch made meanwhile stands.
          if (state.activeConnectionId === id) {
            state.activeConnectionId = older.id;
          }
        }
        Object.assign(target, {
          teamUuid: uuid ?? target.teamUuid ?? null,
          teamName: name ?? target.teamName ?? null,
          email: email ?? target.email ?? null,
        });
        return true;
      });
      if (replaced) onReplacedGrant?.(replaced);
      return target ? toMetadata(target) : this.getActive();
    },

    setActive(id) {
      let found = false;
      mutate(() => {
        found = Boolean(find(id));
        if (!found) return false;
        state.activeConnectionId = id;
        return true;
      });
      return found ? this.getActive() : null;
    },

    remove(id) {
      mutate(() => {
        forgetToken(id);
        state.connections = state.connections.filter(
          (connection) => connection.id !== id,
        );
        if (state.activeConnectionId === id) {
          state.activeConnectionId = state.connections[0]?.id ?? null;
        }
      });
      return this.getActive();
    },

    /** Test seam: drop cached state and re-read from disk. */
    reload() {
      state = load();
    },
  };
}
