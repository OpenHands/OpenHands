/**
 * Where the Canvas host keeps DigitalOcean secrets: the operating system's
 * keychain when it has one, else a file encrypted with the host's
 * `OH_SECRET_KEY`, else this process's memory.
 *
 * The keychain backends drive the operating system's own tools (`security` on
 * macOS, `secret-tool` on Linux desktops) instead of a native Node module, so
 * there is nothing extra to package for the desktop app or the Docker image.
 * On macOS the Keychain item then trusts the system tool rather than the Node
 * binary, so a Node upgrade does not ask for Keychain access again. Secrets
 * reach the tools on stdin, never on the command line, where other processes
 * could read them.
 *
 * Every call is synchronous. Secrets are read once at startup and written on
 * sign-in and sign-out, never per request, and a synchronous store keeps the
 * credential store's API as it was.
 *
 * Nothing here writes to probe a backend. Which one is used depends only on
 * the platform, the tool being installed and `OH_CANVAS_SECRET_STORE`, so
 * opening a store (in a test run on a developer's Mac, say) leaves the real
 * keychain untouched until something is actually saved.
 */

import { execFileSync } from "node:child_process";
import {
  createCipheriv,
  createDecipheriv,
  hkdfSync,
  randomBytes,
} from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import { hostname } from "node:os";
import path from "node:path";

export const SECRET_STORE_KEYCHAIN = "keychain";
export const SECRET_STORE_FILE = "encrypted-file";
export const SECRET_STORE_MEMORY = "memory";

/** Keychain service name; no spaces, so it needs no quoting for `security -i`. */
export const KEYCHAIN_SERVICE = "openhands-agent-canvas";
const KEYCHAIN_LABEL = "OpenHands Agent Canvas";

const SECRETS_FILENAME = "secrets.json";
const SECRETS_FILE_VERSION = 1;
const MACOS_SECURITY = "/usr/bin/security";
/** `security` exit status for "the item could not be found in the keychain". */
const MACOS_ITEM_NOT_FOUND = 44;
const TOOL_TIMEOUT_MS = 10_000;

/** `OH_CANVAS_SECRET_STORE` values. */
const FORCED_KINDS = {
  keychain: SECRET_STORE_KEYCHAIN,
  file: SECRET_STORE_FILE,
  [SECRET_STORE_FILE]: SECRET_STORE_FILE,
  memory: SECRET_STORE_MEMORY,
};

/**
 * @typedef {object} SecretStore
 * @property {string} kind One of the SECRET_STORE_* constants
 * @property {string} label Shown to people: "macOS Keychain", "encrypted file", …
 * @property {boolean} isPersistent Whether secrets survive a restart
 * @property {(account: string) => string | null} read Null when nothing is stored
 * @property {(account: string, value: string) => void} write Throws when the store refuses
 * @property {(account: string) => void} remove Quiet when nothing is stored
 */

function defaultRun(command, args, { input } = {}) {
  return execFileSync(command, args, {
    input,
    encoding: "utf8",
    timeout: TOOL_TIMEOUT_MS,
    stdio: ["pipe", "pipe", "pipe"],
  });
}

function toolFailure(error) {
  const stderr = error?.stderr?.toString?.().trim();
  return stderr || error?.message || String(error);
}

/**
 * macOS Keychain through `/usr/bin/security`.
 *
 * Writing goes through interactive mode (`security -i`), which reads the
 * command, secret included, from stdin. That mode does not report a failed
 * command in its exit status, so every write is read back to confirm it.
 * The value is hex-encoded for `-X` and base64 underneath, so it never needs
 * quoting and `-w` always prints it as plain text.
 */
export function createMacKeychainStore({ run = defaultRun } = {}) {
  function read(account) {
    try {
      const out = run(MACOS_SECURITY, [
        "find-generic-password",
        "-s",
        KEYCHAIN_SERVICE,
        "-a",
        account,
        "-w",
      ]);
      return Buffer.from(out.trim(), "base64").toString("utf8");
    } catch (error) {
      if (error?.status === MACOS_ITEM_NOT_FOUND) return null;
      throw new Error(`macOS Keychain refused to read: ${toolFailure(error)}`);
    }
  }

  return {
    kind: SECRET_STORE_KEYCHAIN,
    label: "macOS Keychain",
    isPersistent: true,
    read,
    write(account, value) {
      const encoded = Buffer.from(value, "utf8").toString("base64");
      const hex = Buffer.from(encoded, "utf8").toString("hex");
      try {
        run(MACOS_SECURITY, ["-i"], {
          input: `add-generic-password -U -s ${KEYCHAIN_SERVICE} -a ${account} -X ${hex}\n`,
        });
      } catch (error) {
        throw new Error(
          `macOS Keychain refused to save: ${toolFailure(error)}`,
        );
      }
      if (read(account) !== value) {
        throw new Error(
          "macOS Keychain did not keep the sign-in. Is the keychain locked?",
        );
      }
    },
    remove(account) {
      try {
        run(MACOS_SECURITY, [
          "delete-generic-password",
          "-s",
          KEYCHAIN_SERVICE,
          "-a",
          account,
        ]);
      } catch (error) {
        if (error?.status !== MACOS_ITEM_NOT_FOUND) throw error;
      }
    },
  };
}

/**
 * Linux Secret Service (GNOME Keyring, KWallet) through `secret-tool`, which
 * reads the secret to store from stdin. Values are base64 so a trailing
 * newline, which `lookup` adds only on some systems, can be trimmed safely.
 */
export function createSecretServiceStore({ run = defaultRun } = {}) {
  const attributes = (account) => [
    "service",
    KEYCHAIN_SERVICE,
    "account",
    account,
  ];
  return {
    kind: SECRET_STORE_KEYCHAIN,
    label: "Secret Service",
    isPersistent: true,
    read(account) {
      try {
        const out = run("secret-tool", ["lookup", ...attributes(account)]);
        const trimmed = out.trim();
        return trimmed ? Buffer.from(trimmed, "base64").toString("utf8") : null;
      } catch (error) {
        // `lookup` exits 1, saying nothing, when nothing matches.
        if (error?.status === 1 && !error?.stderr?.toString?.().trim()) {
          return null;
        }
        throw new Error(
          `Secret Service refused to read: ${toolFailure(error)}`,
        );
      }
    },
    write(account, value) {
      try {
        run(
          "secret-tool",
          ["store", `--label=${KEYCHAIN_LABEL}`, ...attributes(account)],
          { input: Buffer.from(value, "utf8").toString("base64") },
        );
      } catch (error) {
        throw new Error(
          `Secret Service refused to save: ${toolFailure(error)}`,
        );
      }
    },
    remove(account) {
      try {
        run("secret-tool", ["clear", ...attributes(account)]);
      } catch {
        // Nothing stored under this account.
      }
    },
  };
}

/**
 * A file of AES-256-GCM entries, keyed from `OH_SECRET_KEY`: the same secret
 * that already protects Canvas settings on this host. It is the store for
 * hosts without a keychain, such as a container, where the key comes from the
 * environment rather than from a file beside this one.
 */
export function createEncryptedFileStore({ dir, secretKey }) {
  if (!dir || !secretKey) {
    throw new Error("An encrypted secret file needs a directory and a key.");
  }
  const filePath = path.join(dir, SECRETS_FILENAME);
  const key = Buffer.from(
    hkdfSync(
      "sha256",
      secretKey,
      "openhands-agent-canvas",
      "mars-secrets/v1",
      32,
    ),
  );

  function load() {
    try {
      const parsed = JSON.parse(readFileSync(filePath, "utf8"));
      return parsed?.version === SECRETS_FILE_VERSION &&
        parsed.entries &&
        typeof parsed.entries === "object"
        ? parsed.entries
        : {};
    } catch {
      return {};
    }
  }

  function save(entries) {
    mkdirSync(dir, { recursive: true, mode: 0o700 });
    const temporary = `${filePath}.${process.pid}.tmp`;
    writeFileSync(
      temporary,
      JSON.stringify({ version: SECRETS_FILE_VERSION, entries }),
      { mode: 0o600 },
    );
    renameSync(temporary, filePath);
  }

  return {
    kind: SECRET_STORE_FILE,
    label: "encrypted file",
    isPersistent: true,
    read(account) {
      const entry = load()[account];
      if (!entry) return null;
      try {
        const decipher = createDecipheriv(
          "aes-256-gcm",
          key,
          Buffer.from(entry.iv, "base64"),
        );
        decipher.setAuthTag(Buffer.from(entry.tag, "base64"));
        return Buffer.concat([
          decipher.update(Buffer.from(entry.data, "base64")),
          decipher.final(),
        ]).toString("utf8");
      } catch {
        // Written under another OH_SECRET_KEY: unreadable, like a missing entry.
        return null;
      }
    },
    write(account, value) {
      const iv = randomBytes(12);
      const cipher = createCipheriv("aes-256-gcm", key, iv);
      const data = Buffer.concat([
        cipher.update(Buffer.from(value, "utf8")),
        cipher.final(),
      ]);
      save({
        ...load(),
        [account]: {
          iv: iv.toString("base64"),
          tag: cipher.getAuthTag().toString("base64"),
          data: data.toString("base64"),
        },
      });
    },
    remove(account) {
      const entries = load();
      if (!(account in entries)) return;
      delete entries[account];
      save(entries);
    },
  };
}

/** This process only: what is left when nothing else is available. */
export function createMemoryStore() {
  const values = new Map();
  return {
    kind: SECRET_STORE_MEMORY,
    label: "memory",
    isPersistent: false,
    read: (account) => values.get(account) ?? null,
    write: (account, value) => values.set(account, value),
    remove: (account) => values.delete(account),
  };
}

function hasSecretTool(env, run) {
  if (!env.DBUS_SESSION_BUS_ADDRESS) return false;
  try {
    run("secret-tool", ["--version"]);
    return true;
  } catch (error) {
    // Present but complaining about the argument still means installed.
    return error?.code !== "ENOENT" && typeof error?.status === "number";
  }
}

function openKeychain({ platform, env, run, exists }) {
  if (platform === "darwin" && exists(MACOS_SECURITY)) {
    return createMacKeychainStore({ run });
  }
  if (platform === "linux" && hasSecretTool(env, run)) {
    return createSecretServiceStore({ run });
  }
  return null;
}

/**
 * Pick the store for this host. `OH_CANVAS_SECRET_STORE=keychain|file|memory`
 * forces one; a forced store that is unavailable gives memory and a warning,
 * never a different persistent store. Otherwise: the keychain, then the
 * encrypted file, then memory.
 *
 * @param {object} [options]
 * @param {Record<string, string | undefined>} [options.env]
 * @param {string} [options.platform]
 * @param {string | null} [options.dir] Directory for the encrypted file
 * @param {string | null} [options.secretKey] Defaults to `env.OH_SECRET_KEY`
 * @param {typeof defaultRun} [options.run] Runs a tool; injectable for tests
 * @param {(file: string) => boolean} [options.exists]
 * @param {(message: string) => void} [options.log]
 * @returns {SecretStore}
 */
export function openSecretStore({
  env = process.env,
  platform = process.platform,
  dir = null,
  secretKey = env.OH_SECRET_KEY?.trim() || null,
  run = defaultRun,
  exists = existsSync,
  log = (message) => console.warn(`[mars-secrets] ${message}`),
} = {}) {
  const requested = env.OH_CANVAS_SECRET_STORE?.trim().toLowerCase() || null;
  const forced = requested ? FORCED_KINDS[requested] : null;
  if (requested && !forced) {
    log(
      `OH_CANVAS_SECRET_STORE=${JSON.stringify(requested)} is not one of keychain, file, memory; choosing automatically.`,
    );
  }
  const file = () =>
    dir && secretKey ? createEncryptedFileStore({ dir, secretKey }) : null;

  if (forced === SECRET_STORE_MEMORY) return createMemoryStore();
  if (forced) {
    const store =
      forced === SECRET_STORE_KEYCHAIN
        ? openKeychain({ platform, env, run, exists })
        : file();
    if (store) return store;
    log(
      `OH_CANVAS_SECRET_STORE=${requested} is not available on this host; keeping sign-ins in memory.`,
    );
    return createMemoryStore();
  }
  return (
    openKeychain({ platform, env, run, exists }) ??
    file() ??
    createMemoryStore()
  );
}

/** What the UI shows about where a sign-in is kept. */
export function describeSecretStore(store, host = hostname()) {
  return { kind: store.kind, label: store.label, host };
}
