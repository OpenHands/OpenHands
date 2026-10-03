/**
 * Encrypted store for DigitalOcean credentials used by the MARS backend.
 *
 * Holds *multiple* named connections rather than a single token. An OAuth
 * access token is pinned to one team (`team_uuid` in the grant) and MARS
 * sessions are team-scoped, so working across teams means holding one
 * credential per team and switching between them.
 *
 * Only the token is encrypted; the surrounding metadata stays readable so the
 * UI can label connections without a decrypt round-trip. Nothing here is ever
 * handed to the renderer except through `toMetadata`, which omits the token.
 */

import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const STORE_FILENAME = "mars-credentials.json";
const STORE_VERSION = 1;

export const CREDENTIAL_KIND_OAUTH = "oauth";
export const CREDENTIAL_KIND_PAT = "pat";

const LEGACY_PAT_LENGTH = 64;
const V1_PAT_LENGTH = 71;
const V1_PAT_PREFIX = "do";

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

function emptyState() {
  return {
    version: STORE_VERSION,
    deviceId: randomUUID(),
    activeConnectionId: null,
    connections: [],
  };
}

/** Public shape of a connection: everything except the secret. */
function toMetadata(connection) {
  if (!connection) return null;
  return {
    id: connection.id,
    kind: connection.kind,
    label: connection.label,
    teamName: connection.teamName ?? null,
    expiresAt: connection.expiresAt ?? null,
    isExpired: isExpired(connection),
  };
}

function isExpired(connection) {
  if (!connection?.expiresAt) return false;
  return Date.parse(connection.expiresAt) <= Date.now();
}

/**
 * @param {object} options
 * @param {string} options.userDataPath
 * @param {Pick<import("electron").SafeStorage, "isEncryptionAvailable" | "encryptString" | "decryptString"> | null} [options.safeStorage]
 *   Omitted in tests, which exercise the unencrypted in-memory path.
 */
export function createCredentialStore({ userDataPath, safeStorage = null }) {
  const filePath = path.join(userDataPath, STORE_FILENAME);

  // When the OS keychain is unavailable (common on bare Linux desktops),
  // credentials live only for this process rather than being written to disk
  // in the clear.
  const canEncrypt = Boolean(safeStorage?.isEncryptionAvailable?.());
  const memoryTokens = new Map();

  let state = load();

  function load() {
    try {
      const parsed = JSON.parse(readFileSync(filePath, "utf8"));
      if (parsed?.version !== STORE_VERSION) return emptyState();
      return {
        ...emptyState(),
        ...parsed,
        connections: Array.isArray(parsed.connections)
          ? parsed.connections
          : [],
      };
    } catch {
      // Missing or corrupt store — start clean rather than blocking sign-in.
      return emptyState();
    }
  }

  function persist() {
    mkdirSync(path.dirname(filePath), { recursive: true });
    const serializable = {
      ...state,
      connections: state.connections.map((connection) => ({
        ...connection,
        // Session-only credentials must not leave a token behind on disk.
        token: canEncrypt ? connection.token : null,
      })),
    };
    writeFileSync(filePath, JSON.stringify(serializable, null, 2), {
      mode: 0o600,
    });
  }

  function encrypt(token) {
    if (!canEncrypt) return null;
    return safeStorage.encryptString(token).toString("base64");
  }

  function decrypt(connection) {
    if (!canEncrypt) return memoryTokens.get(connection.id) ?? null;
    if (!connection.token) return null;
    try {
      return safeStorage.decryptString(Buffer.from(connection.token, "base64"));
    } catch {
      // Keychain rotated or the profile moved between machines.
      return null;
    }
  }

  function find(id) {
    return state.connections.find((connection) => connection.id === id) ?? null;
  }

  return {
    /** True when tokens survive a restart. Surfaced so the UI can warn. */
    get isPersistent() {
      return canEncrypt;
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
      const connection = find(state.activeConnectionId);
      if (!connection || isExpired(connection)) return null;
      return decrypt(connection);
    },

    /**
     * @param {object} input
     * @param {string} input.kind
     * @param {string} input.token
     * @param {string | null} [input.expiresAt]
     * @param {string} input.label
     * @param {string | null} [input.teamName]
     */
    save({ kind, token, expiresAt = null, label, teamName = null }) {
      const id = randomUUID();
      const connection = {
        id,
        kind,
        label,
        teamName,
        expiresAt,
        token: encrypt(token),
      };
      if (!canEncrypt) memoryTokens.set(id, token);

      state.connections = [...state.connections, connection];
      state.activeConnectionId = id;
      persist();
      return toMetadata(connection);
    },

    setActive(id) {
      if (!find(id)) return null;
      state.activeConnectionId = id;
      persist();
      return this.getActive();
    },

    remove(id) {
      memoryTokens.delete(id);
      state.connections = state.connections.filter(
        (connection) => connection.id !== id,
      );
      if (state.activeConnectionId === id) {
        state.activeConnectionId = state.connections[0]?.id ?? null;
      }
      persist();
      return this.getActive();
    },

    /** Test seam: drop cached state and re-read from disk. */
    reload() {
      state = load();
    },
  };
}
