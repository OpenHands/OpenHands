import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { BACKENDS_STORAGE_KEY } from "#/api/backend-registry/storage";
import { hadPairingFailure, redeemPairingLink } from "#/api/mars/pairing";

type StoredBackend = { id: string; host: string; apiKey: string; kind: string };

const storedBackends = (): StoredBackend[] =>
  JSON.parse(window.localStorage.getItem(BACKENDS_STORAGE_KEY) ?? "[]");

const redeemed = (key = "canvas-key") =>
  vi.fn(async () => Response.json({ result: { sessionApiKey: key } }));

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
});

afterEach(() => {
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_BASE_PATH__;
  window.history.replaceState(null, "", "/");
});

describe("redeemPairingLink", () => {
  it("leaves every other page alone", async () => {
    window.history.replaceState(null, "", "/conversations#not-a-secret");
    const fetchImpl = redeemed();

    expect(await redeemPairingLink(fetchImpl)).toBe(false);
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(window.location.hash).toBe("#not-a-secret");
  });

  it("lets this device in with the key the link redeems, and clears the address bar", async () => {
    window.history.replaceState(null, "", "/pair#s3cret");
    const fetchImpl = redeemed();

    const paired = await redeemPairingLink(fetchImpl);

    expect(paired).toBe(true);
    expect(fetchImpl).toHaveBeenCalledWith(
      "/mars/pair",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ secret: "s3cret" }),
      }),
    );
    expect(`${window.location.pathname}${window.location.hash}`).toBe("/");
    const local = storedBackends().filter(
      (backend) => backend.host === window.location.origin,
    );
    expect(local).toEqual([
      expect.objectContaining({ kind: "local", apiKey: "canvas-key" }),
    ]);
  });

  it("updates the backend this address already has instead of adding one", async () => {
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([
        {
          id: "mac",
          name: "Mac",
          host: window.location.origin,
          apiKey: "stale",
          kind: "local",
        },
      ]),
    );
    window.history.replaceState(null, "", "/pair#s3cret");

    await redeemPairingLink(redeemed());

    expect(storedBackends()).toEqual([
      expect.objectContaining({ id: "mac", apiKey: "canvas-key" }),
    ]);
  });

  it("works under the base path Canvas is mounted at", async () => {
    (window as unknown as Record<string, unknown>).__AGENT_CANVAS_BASE_PATH__ =
      "/canvas";
    window.history.replaceState(null, "", "/canvas/pair#s3cret");
    const fetchImpl = redeemed();

    expect(await redeemPairingLink(fetchImpl)).toBe(true);
    expect(fetchImpl).toHaveBeenCalledWith("/mars/pair", expect.anything());
    expect(`${window.location.pathname}${window.location.hash}`).toBe(
      "/canvas/",
    );
  });

  it("leaves a DigitalOcean session's backend on this address alone", async () => {
    const session = {
      id: "do-session",
      name: "Agent",
      host: `${window.location.origin}/mars/sessions/s1`,
      apiKey: "guest-key",
      kind: "local",
      marsSessionId: "s1",
    };
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([session]),
    );
    window.history.replaceState(null, "", "/pair#s3cret");

    await redeemPairingLink(redeemed());

    expect(storedBackends()).toEqual([
      session,
      expect.objectContaining({
        host: window.location.origin,
        apiKey: "canvas-key",
      }),
    ]);
  });

  it("names a new backend on a plain HTTP address, which has no crypto.randomUUID", async () => {
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([
        {
          id: "default-local",
          name: "Remote",
          host: "https://remote.openhands.dev",
          apiKey: "remote-key",
          kind: "local",
        },
      ]),
    );
    window.history.replaceState(null, "", "/pair#s3cret");
    Object.defineProperty(crypto, "randomUUID", {
      value: undefined,
      configurable: true,
    });
    try {
      expect(await redeemPairingLink(redeemed())).toBe(true);
    } finally {
      delete (crypto as unknown as Record<string, unknown>).randomUUID;
    }

    const added = storedBackends().find(
      (backend) => backend.host === window.location.origin,
    );
    expect(added?.id).toMatch(/^backend-/);
    expect(typeof crypto.randomUUID).toBe("function");
  });

  it("notes a link that did not work, for the key screen", async () => {
    window.history.replaceState(null, "", "/pair#used");
    const fetchImpl = vi.fn(async () =>
      Response.json({ error: { message: "used" } }, { status: 400 }),
    );

    expect(await redeemPairingLink(fetchImpl)).toBe(false);
    expect(hadPairingFailure()).toBe(true);
    expect(`${window.location.pathname}${window.location.hash}`).toBe("/");
  });
});
