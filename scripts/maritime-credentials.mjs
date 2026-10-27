/**
 * Where the Maritime provider keeps its secrets: the user's Maritime API key,
 * and the session API key of each agent-server Canvas created on Maritime.
 *
 * The session keys are kept because Maritime stores an agent's secret env
 * vars encrypted and only ever shows them masked, so the key Canvas generated
 * at create time cannot be read back from Maritime later. Losing it is not
 * fatal (the bridge can set a new one, see `resetAccessKey`), it only costs a
 * restart of that agent.
 *
 * Storage follows the DigitalOcean credential store: the Agent Canvas server
 * hands in a `SecretStore` (scripts/mars-secret-store.mjs: the OS keychain, a
 * file encrypted with `OH_SECRET_KEY`, or memory), and the desktop app passes
 * Electron's `safeStorage`, in which case one encrypted blob is written to
 * `maritime-credentials.json`. With neither, secrets last for this process.
 * Nothing here ever reaches the renderer.
 */

import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { hostname, platform } from "node:os";
import path from "node:path";

const STORE_FILENAME = "maritime-credentials.json";
const STORE_VERSION = 1;
/** Account name the record is kept under in a `SecretStore`. */
export const MARITIME_SECRET_ACCOUNT = "maritime/credentials";
const RECORD_VERSION = 1;

function emptyRecord() {
  return { apiKey: null, sessionKeys: {} };
}

/** The record in a stored string; an empty one for anything unreadable. */
function parseRecord(serialized) {
  try {
    const record = JSON.parse(serialized);
    if (record?.v !== RECORD_VERSION) return emptyRecord();
    return {
      apiKey: typeof record.apiKey === "string" ? record.apiKey : null,
      sessionKeys:
        record.sessionKeys && typeof record.sessionKeys === "object"
          ? Object.fromEntries(
              Object.entries(record.sessionKeys).filter(
                ([, value]) => typeof value === "string",
              ),
            )
          : {},
    };
  } catch {
    return emptyRecord();
  }
}

/**
 * @param {object} [options]
 * @param {string | null} [options.userDataPath] Where the desktop app's
 *   encrypted file lives; null keeps everything in memory.
 * @param {Pick<import("electron").SafeStorage, "isEncryptionAvailable" | "encryptString" | "decryptString"> | null} [options.safeStorage]
 * @param {import("./mars-secret-store.mjs").SecretStore | null} [options.secrets]
 * @param {(message: string) => void} [options.log]
 */
export function createMaritimeCredentialStore({
  userDataPath = null,
  safeStorage = null,
  secrets = null,
  log = (message) => console.warn(`[maritime-credentials] ${message}`),
} = {}) {
  const filePath = userDataPath
    ? path.join(userDataPath, STORE_FILENAME)
    : null;
  const canEncrypt = Boolean(
    !secrets && filePath && safeStorage?.isEncryptionAvailable?.(),
  );
  /** False once the store refused a write; the UI says it was not saved. */
  let isSaved = true;
  let record = load();

  function load() {
    if (secrets) {
      try {
        const stored = secrets.read(MARITIME_SECRET_ACCOUNT);
        return stored ? parseRecord(stored) : emptyRecord();
      } catch (error) {
        // A locked keychain: start signed out rather than failing to start.
        log(error instanceof Error ? error.message : String(error));
        return emptyRecord();
      }
    }
    if (!canEncrypt) return emptyRecord();
    try {
      const file = JSON.parse(readFileSync(filePath, "utf8"));
      if (file?.version !== STORE_VERSION || typeof file.blob !== "string") {
        return emptyRecord();
      }
      return parseRecord(
        safeStorage.decryptString(Buffer.from(file.blob, "base64")),
      );
    } catch {
      // Missing file, or a keychain that rotated: sign in again.
      return emptyRecord();
    }
  }

  function persist() {
    const serialized = JSON.stringify({ v: RECORD_VERSION, ...record });
    try {
      if (secrets) {
        if (record.apiKey === null && !Object.keys(record.sessionKeys).length) {
          secrets.remove(MARITIME_SECRET_ACCOUNT);
        } else {
          secrets.write(MARITIME_SECRET_ACCOUNT, serialized);
        }
        isSaved = true;
        return;
      }
      if (!canEncrypt) {
        isSaved = false;
        return;
      }
      mkdirSync(path.dirname(filePath), { recursive: true });
      const blob = safeStorage.encryptString(serialized).toString("base64");
      const temporary = `${filePath}.${process.pid}.tmp`;
      writeFileSync(
        temporary,
        JSON.stringify({ version: STORE_VERSION, blob }, null, 2),
        { mode: 0o600 },
      );
      renameSync(temporary, filePath);
      isSaved = true;
    } catch (error) {
      // Usable for this run; the UI says it was not saved.
      isSaved = false;
      log(error instanceof Error ? error.message : String(error));
    }
  }

  /** Where secrets are kept, for the UI. Mirrors the DigitalOcean store. */
  function storage() {
    const host = hostname();
    if (secrets) {
      return {
        kind: secrets.isPersistent ? secrets.kind : "memory",
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
    get storage() {
      return storage();
    },

    /** False when the last write was refused or nothing persists secrets. */
    get isSaved() {
      return isSaved && storage().kind !== "memory";
    },

    getApiKey() {
      return record.apiKey;
    },

    /** Replaces the key. Session keys stay: they belong to the agents. */
    saveApiKey(apiKey) {
      record = { ...record, apiKey };
      persist();
    },

    getSessionKey(agentId) {
      return record.sessionKeys[agentId] ?? null;
    },

    saveSessionKey(agentId, sessionKey) {
      record = {
        ...record,
        sessionKeys: { ...record.sessionKeys, [agentId]: sessionKey },
      };
      persist();
    },

    forgetSessionKey(agentId) {
      if (!(agentId in record.sessionKeys)) return;
      const { [agentId]: _removed, ...rest } = record.sessionKeys;
      record = { ...record, sessionKeys: rest };
      persist();
    },

    /**
     * Sign out: drops the API key. Session keys stay, like the backends that
     * already hold them: they open only their own agent-server, and signing
     * back in should not cost each agent a restart.
     */
    forgetApiKey() {
      record = { ...record, apiKey: null };
      persist();
    },

    /** Test seam: drop cached state and re-read from storage. */
    reload() {
      record = load();
    },
  };
}
