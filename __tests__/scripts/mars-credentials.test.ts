// @vitest-environment node
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
  CREDENTIAL_KIND_OAUTH,
  CREDENTIAL_KIND_PAT,
  createCredentialStore,
  isValidPatFormat,
} from "../../scripts/mars-credentials.mjs";
import {
  createEncryptedFileStore,
  createMemoryStore,
} from "../../scripts/mars-secret-store.mjs";

const STORE_FILENAME = "mars-credentials.json";

/** Stand-in for Electron's safeStorage; reversible, not secure. */
const fakeSafeStorage = {
  isEncryptionAvailable: () => true,
  encryptString: (value: string) => Buffer.from(`enc:${value}`),
  decryptString: (buffer: Buffer) => buffer.toString().replace(/^enc:/, ""),
};

function newStore({ safeStorage = fakeSafeStorage } = {}) {
  const userDataPath = mkdtempSync(join(tmpdir(), "mars-creds-"));
  return {
    userDataPath,
    store: createCredentialStore({ userDataPath, safeStorage }),
  };
}

describe("isValidPatFormat", () => {
  it.each([
    ["a".repeat(64), true],
    [`do${"a".repeat(69)}`, true],
    [`xx${"a".repeat(69)}`, false],
    ["short", false],
    ["", false],
  ])("validates %s as %s the way doctl does", (token, expected) => {
    expect(isValidPatFormat(token)).toBe(expected);
  });
});

describe("createCredentialStore", () => {
  it("keeps the token out of the metadata it hands to the renderer", () => {
    const { store } = newStore();

    const metadata = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: "a".repeat(64),
      label: "Platform team",
    });

    expect(metadata).not.toHaveProperty("token");
    expect(JSON.stringify(store.list())).not.toContain("a".repeat(64));
    expect(store.getActiveToken()).toBe("a".repeat(64));
  });

  it("holds one credential per team and switches between them", () => {
    const { store } = newStore();

    const first = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: "1".repeat(64),
      label: "Team one",
    });
    const second = store.save({
      kind: CREDENTIAL_KIND_OAUTH,
      token: "2".repeat(64),
      label: "Team two",
    });

    expect(store.getActive()?.id).toBe(second?.id);

    store.setActive(first!.id);
    expect(store.getActiveToken()).toBe("1".repeat(64));
  });

  it("survives a restart by reading the encrypted file back", () => {
    const { userDataPath, store } = newStore();
    store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: "3".repeat(64),
      label: "Persisted",
    });

    const reopened = createCredentialStore({
      userDataPath,
      safeStorage: fakeSafeStorage,
    });

    expect(reopened.getActiveToken()).toBe("3".repeat(64));
    expect(reopened.isPersistent).toBe(true);
  });

  it("treats an expired credential as unusable without deleting it", () => {
    const { store } = newStore();
    store.save({
      kind: CREDENTIAL_KIND_OAUTH,
      token: "4".repeat(64),
      label: "Lapsed",
      expiresAt: new Date(Date.now() - 1000).toISOString(),
    });

    // Implicit grants cannot be refreshed, so the UI prompts for a fresh
    // sign-in rather than silently discarding the connection.
    expect(store.getActive()?.isExpired).toBe(true);
    expect(store.getActiveToken()).toBeNull();
    expect(store.list()).toHaveLength(1);
  });

  it("never writes a plaintext token when the keychain is unavailable", () => {
    const { userDataPath, store } = newStore({
      safeStorage: { ...fakeSafeStorage, isEncryptionAvailable: () => false },
    });

    store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: "5".repeat(64),
      label: "Session only",
    });

    const onDisk = readFileSync(join(userDataPath, STORE_FILENAME), "utf8");
    expect(onDisk).not.toContain("5".repeat(64));
    // Usable for this process, but gone after a restart.
    expect(store.getActiveToken()).toBe("5".repeat(64));
    expect(store.isPersistent).toBe(false);
  });

  it("decrypts each token once, not on every use", () => {
    let decrypts = 0;
    const { userDataPath, store } = newStore();
    store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: "8".repeat(64),
      label: "Cached",
    });
    const reopened = createCredentialStore({
      userDataPath,
      safeStorage: {
        ...fakeSafeStorage,
        decryptString: (buffer: Buffer) => {
          decrypts += 1;
          return fakeSafeStorage.decryptString(buffer);
        },
      },
    });

    expect(reopened.getActiveToken()).toBe("8".repeat(64));
    expect(reopened.getActiveToken()).toBe("8".repeat(64));
    expect(decrypts).toBe(1);

    reopened.remove(reopened.getActive()!.id);
    expect(reopened.getActiveToken()).toBeNull();
  });

  it("keeps everything in memory without a userDataPath", () => {
    const store = createCredentialStore({
      userDataPath: null,
      safeStorage: fakeSafeStorage,
    });

    store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: "9".repeat(64),
      label: "Web",
    });

    expect(store.getActiveToken()).toBe("9".repeat(64));
    expect(store.isPersistent).toBe(false);
  });

  it("falls back to another connection when the active one is removed", () => {
    const { store } = newStore();
    const first = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: "6".repeat(64),
      label: "Team one",
    });
    const second = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: "7".repeat(64),
      label: "Team two",
    });

    expect(store.remove(second!.id)?.id).toBe(first?.id);
  });
});

describe("createCredentialStore with a SecretStore (the Agent Canvas server)", () => {
  const TOKEN = `dop_v1_${"d".repeat(64)}`;

  function serverStore(dir = mkdtempSync(join(tmpdir(), "mars-server-"))) {
    const secrets = createEncryptedFileStore({ dir, secretKey: "host-key" });
    return {
      dir,
      secrets,
      store: createCredentialStore({
        userDataPath: dir,
        secrets,
        log: () => {},
      }),
    };
  }

  it("keeps a sign-in across a restart, with no token in the metadata file", () => {
    const { dir, store } = serverStore();
    store.save({ kind: CREDENTIAL_KIND_PAT, token: TOKEN, label: "Token" });

    const restarted = serverStore(dir).store;

    expect(restarted.getActiveToken()).toBe(TOKEN);
    expect(restarted.getActive()).toMatchObject({ isExpired: false });
    expect(restarted.storage).toMatchObject({ kind: "encrypted-file" });
    expect(readFileSync(join(dir, STORE_FILENAME), "utf8")).not.toContain(
      "d".repeat(64),
    );
  });

  it("keeps one connection when the same token is entered again", () => {
    const { store } = serverStore();

    const first = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: TOKEN,
      label: "A",
    });
    const again = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: TOKEN,
      label: "A",
    });

    expect(again.id).toBe(first.id);
    expect(again.reused).toBe(true);
    expect(store.list()).toHaveLength(1);
  });

  it("gives a team's new sign-in the id the team already had", () => {
    const { store } = serverStore();
    const first = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: TOKEN,
      label: "A",
    });
    store.setTeam(first.id, { uuid: "team-1", name: "My Team" });
    const second = store.save({
      kind: CREDENTIAL_KIND_OAUTH,
      token: "oauth-access-token",
      label: "DigitalOcean",
    });

    const merged = store.setTeam(second.id, {
      uuid: "team-1",
      name: "My Team",
    });

    expect(merged).toMatchObject({
      id: first.id,
      kind: "oauth",
      teamName: "My Team",
    });
    expect(store.list()).toHaveLength(1);
    expect(store.getActiveToken()).toBe("oauth-access-token");
  });

  it("leaves a team switch made while a new sign-in was being checked", () => {
    const { store } = serverStore();
    const teamOne = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: TOKEN,
      label: "A",
    });
    store.setTeam(teamOne.id, { uuid: "team-1", name: "Team One" });
    const teamTwo = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: "token-two",
      label: "B",
    });
    store.setTeam(teamTwo.id, { uuid: "team-2", name: "Team Two" });
    const again = store.save({
      kind: CREDENTIAL_KIND_OAUTH,
      token: "oauth-access-token",
      label: "DigitalOcean",
    });
    store.setActive(teamTwo.id);

    store.setTeam(again.id, { uuid: "team-1", name: "Team One" });

    expect(store.getActive()?.id).toBe(teamTwo.id);
    expect(store.getToken(teamOne.id)).toBe("oauth-access-token");
  });

  it("keeps a versioned record in the store, not a bare token", () => {
    const { store, secrets } = serverStore();
    const saved = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: TOKEN,
      label: "A",
    });

    expect(JSON.parse(secrets.read(`digitalocean/${saved.id}`)!)).toEqual({
      v: 1,
      kind: "pat",
      token: TOKEN,
    });
  });

  it("asks for sign-in again when a saved secret can no longer be read", () => {
    const { dir, store, secrets } = serverStore();
    const saved = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: TOKEN,
      label: "A",
    });
    secrets.remove(`digitalocean/${saved.id}`);

    const restarted = serverStore(dir).store;

    expect(restarted.getActive()).toMatchObject({
      id: saved.id,
      isExpired: true,
    });
  });

  it("keeps a sign-in for this run when the store refuses it, and says it was not saved", () => {
    const refusing = {
      ...createMemoryStore(),
      isPersistent: true,
      write: () => {
        throw new Error("macOS Keychain refused to save");
      },
    };
    const store = createCredentialStore({
      userDataPath: mkdtempSync(join(tmpdir(), "mars-refused-")),
      secrets: refusing,
      log: () => {},
    });

    const saved = store.save({
      kind: CREDENTIAL_KIND_PAT,
      token: TOKEN,
      label: "A",
    });

    expect(saved).toMatchObject({ isSaved: false, isExpired: false });
    expect(store.getActiveToken()).toBe(TOKEN);
  });
});
