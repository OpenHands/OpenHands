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
