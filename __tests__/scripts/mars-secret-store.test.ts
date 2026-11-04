// @vitest-environment node
import { mkdtempSync, readFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

import {
  KEYCHAIN_SERVICE,
  SECRET_STORE_FILE,
  SECRET_STORE_KEYCHAIN,
  SECRET_STORE_MEMORY,
  createEncryptedFileStore,
  createMacKeychainStore,
  createSecretServiceStore,
  openSecretStore,
} from "../../scripts/mars-secret-store.mjs";

const SECRET = '{"token":"dop_v1_secret"}';

type Call = { command: string; args: string[]; input?: string };

/** Fails like execFileSync: an error carrying the exit status. */
function exitWith(status: number, stderr = "") {
  return Object.assign(new Error(`exit ${status}`), { status, stderr });
}

/** An in-memory stand-in for the `security` tool's generic passwords. */
function fakeSecurity() {
  const items = new Map<string, string>();
  const calls: Call[] = [];
  const run = vi.fn(
    (command: string, args: string[], options?: { input?: string }) => {
      calls.push({ command, args, input: options?.input });
      if (args[0] === "-i") {
        const words = options!.input!.trim().split(" ");
        const value = (flag: string) => words[words.indexOf(flag) + 1];
        items.set(
          `${value("-s")}/${value("-a")}`,
          Buffer.from(value("-X"), "hex").toString(),
        );
        return "";
      }
      const account = args[args.indexOf("-a") + 1];
      const key = `${args[args.indexOf("-s") + 1]}/${account}`;
      if (args[0] === "find-generic-password") {
        if (!items.has(key)) throw exitWith(44);
        return `${items.get(key)}\n`;
      }
      if (args[0] === "delete-generic-password") {
        if (!items.delete(key)) throw exitWith(44);
        return "";
      }
      throw new Error(`unexpected ${args.join(" ")}`);
    },
  );
  return { run, calls, items };
}

describe("macOS Keychain store", () => {
  it("saves through stdin, never on the command line, and reads it back", () => {
    const security = fakeSecurity();
    const store = createMacKeychainStore({ run: security.run });

    store.write("digitalocean/c1", SECRET);

    expect(store.read("digitalocean/c1")).toBe(SECRET);
    for (const call of security.calls) {
      expect(call.args.join(" ")).not.toContain("dop_v1_secret");
    }
    expect(security.calls[0].input).toContain(`-s ${KEYCHAIN_SERVICE}`);
    expect(store).toMatchObject({
      kind: SECRET_STORE_KEYCHAIN,
      label: "macOS Keychain",
      isPersistent: true,
    });
  });

  it("reads a missing item as nothing, and removes quietly", () => {
    const store = createMacKeychainStore({ run: fakeSecurity().run });

    expect(store.read("digitalocean/none")).toBeNull();
    expect(() => store.remove("digitalocean/none")).not.toThrow();
  });

  it("refuses a save the keychain did not keep, such as a locked one", () => {
    const security = fakeSecurity();
    // Interactive mode exits 0 even when its command failed.
    security.run.mockImplementationOnce(() => "");
    const store = createMacKeychainStore({ run: security.run });

    expect(() => store.write("digitalocean/c1", SECRET)).toThrow(/locked/);
  });
});

describe("Secret Service store", () => {
  it("passes the secret on stdin and reads it back", () => {
    const stored = new Map<string, string>();
    const run = vi.fn(
      (_command: string, args: string[], options?: { input?: string }) => {
        const account = args[args.indexOf("account") + 1];
        if (args[0] === "store") {
          stored.set(account, options!.input!);
          return "";
        }
        if (args[0] === "lookup") {
          if (!stored.has(account)) throw exitWith(1);
          return `${stored.get(account)}\n`;
        }
        return "";
      },
    );
    const store = createSecretServiceStore({ run });

    store.write("digitalocean/c1", SECRET);

    expect(store.read("digitalocean/c1")).toBe(SECRET);
    expect(store.read("digitalocean/other")).toBeNull();
    expect(JSON.stringify(run.mock.calls.map((call) => call[1]))).not.toContain(
      "dop_v1_secret",
    );
  });
});

describe("encrypted file store", () => {
  it("round-trips without writing the secret in the clear, readable only by its owner", () => {
    const dir = mkdtempSync(join(tmpdir(), "mars-secrets-"));
    const store = createEncryptedFileStore({ dir, secretKey: "host-key" });

    store.write("digitalocean/c1", SECRET);

    const reopened = createEncryptedFileStore({ dir, secretKey: "host-key" });
    expect(reopened.read("digitalocean/c1")).toBe(SECRET);
    expect(readFileSync(join(dir, "secrets.json"), "utf8")).not.toContain(
      "dop_v1_secret",
    );
    expect(statSync(join(dir, "secrets.json")).mode & 0o777).toBe(0o600);
  });

  it("reads nothing under a different key, and forgets on remove", () => {
    const dir = mkdtempSync(join(tmpdir(), "mars-secrets-"));
    const store = createEncryptedFileStore({ dir, secretKey: "host-key" });
    store.write("digitalocean/c1", SECRET);

    expect(
      createEncryptedFileStore({ dir, secretKey: "other-key" }).read(
        "digitalocean/c1",
      ),
    ).toBeNull();
    store.remove("digitalocean/c1");
    expect(store.read("digitalocean/c1")).toBeNull();
  });
});

describe("openSecretStore", () => {
  const dir = mkdtempSync(join(tmpdir(), "mars-open-"));
  const missingTool = vi.fn(() => {
    throw Object.assign(new Error("not found"), { code: "ENOENT" });
  });

  it("uses the macOS Keychain on a Mac, without writing anything to choose it", () => {
    const security = fakeSecurity();

    const store = openSecretStore({
      env: {},
      platform: "darwin",
      dir,
      run: security.run,
      exists: () => true,
    });

    expect(store.label).toBe("macOS Keychain");
    expect(security.calls).toEqual([]);
  });

  it("falls back to the encrypted file on a host without a keychain, then to memory", () => {
    const withKey = openSecretStore({
      env: { OH_SECRET_KEY: "host-key" },
      platform: "linux",
      dir,
      run: missingTool,
    });
    const withoutKey = openSecretStore({
      env: {},
      platform: "linux",
      dir,
      run: missingTool,
    });

    expect(withKey.kind).toBe(SECRET_STORE_FILE);
    expect(withoutKey.kind).toBe(SECRET_STORE_MEMORY);
  });

  it("honours OH_CANVAS_SECRET_STORE, and never swaps a forced store for another one", () => {
    const logs: string[] = [];

    const forcedFile = openSecretStore({
      env: { OH_CANVAS_SECRET_STORE: "file", OH_SECRET_KEY: "host-key" },
      platform: "darwin",
      dir,
      exists: () => true,
    });
    const forcedKeychain = openSecretStore({
      env: { OH_CANVAS_SECRET_STORE: "keychain", OH_SECRET_KEY: "host-key" },
      platform: "linux",
      dir,
      run: missingTool,
      log: (message) => logs.push(message),
    });

    expect(forcedFile.kind).toBe(SECRET_STORE_FILE);
    expect(forcedKeychain.kind).toBe(SECRET_STORE_MEMORY);
    expect(logs[0]).toMatch(/not available/);
  });
});
