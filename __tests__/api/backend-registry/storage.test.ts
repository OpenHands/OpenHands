import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ACTIVE_BACKEND_STORAGE_KEY,
  BACKENDS_STORAGE_KEY,
  readStoredActiveBackend,
  readStoredBackends,
  writeStoredActiveBackend,
  writeStoredBackends,
} from "#/api/backend-registry/storage";
import { SESSION_KEYS_STORAGE_KEY } from "#/api/backend-registry/session-key-store";
import type { Backend } from "#/api/backend-registry/types";

const ORIGINAL_LOCATION = window.location;

function mockWindowLocation(url: string) {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: new URL(url),
  });
}

afterEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_LOCK_TO_CLOUD__;
  Object.defineProperty(window, "location", {
    configurable: true,
    value: ORIGINAL_LOCATION,
  });
  window.history.pushState({}, "", "/");
  vi.unstubAllEnvs();
});

describe("backend-registry storage", () => {
  it("round-trips a list of backends", () => {
    const backends: Backend[] = [
      {
        id: "abc",
        name: "Local 1",
        host: "http://127.0.0.1:9000",
        apiKey: "key-1",
        kind: "local",
      },
      {
        id: "xyz",
        name: "Production",
        host: "https://app.all-hands.dev",
        apiKey: "bearer-2",
        kind: "cloud",
      },
    ];

    writeStoredBackends(backends);

    expect(readStoredBackends()).toEqual(backends);
  });

  it("returns empty list when storage is malformed", () => {
    window.localStorage.setItem(BACKENDS_STORAGE_KEY, "{not-json");
    expect(readStoredBackends()).toEqual([]);
  });

  it("does not seed the default Local backend when launcher details are missing", () => {
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    expect(window.localStorage.getItem(BACKENDS_STORAGE_KEY)).toBeNull();

    expect(readStoredBackends()).toEqual([]);
    expect(window.localStorage.getItem(BACKENDS_STORAGE_KEY)).toBeNull();
  });

  it("seeds the default Local backend when host and API key are available", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
    vi.stubEnv("VITE_SESSION_API_KEY", "fresh-session-key");

    const result = readStoredBackends();

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      id: "default-local",
      host: "http://localhost:9000",
      apiKey: "fresh-session-key",
      kind: "local",
    });
    expect(window.localStorage.getItem(BACKENDS_STORAGE_KEY)).not.toBeNull();
    expect(readStoredBackends()).toEqual(result);
  });

  it("seeds a cookie-auth Cloud backend when locked to the current origin", () => {
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_LOCK_TO_CLOUD__ = window.location.origin;
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([
        {
          id: "stale-local",
          name: "Stale Local",
          host: "http://localhost:18000",
          apiKey: "stale-session",
          kind: "local",
        },
      ]),
    );

    const result = readStoredBackends();

    expect(result).toEqual([
      {
        id: "locked-cloud",
        name: "OpenHands Cloud",
        host: window.location.origin,
        apiKey: "",
        kind: "cloud",
        authMode: "cookie",
      },
    ]);
    expect(readStoredActiveBackend()).toEqual({
      backendId: "locked-cloud",
      orgId: null,
    });
  });

  it("seeds the cookie-auth Cloud backend on the current origin when the locked host is an equivalent transition domain", () => {
    mockWindowLocation("https://pr-254.staging.openhands.dev/canvas");
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_LOCK_TO_CLOUD__ = "https://pr-254.staging.all-hands.dev";

    expect(readStoredBackends()).toEqual([
      {
        id: "locked-cloud",
        name: "OpenHands Cloud",
        host: "https://pr-254.staging.openhands.dev",
        apiKey: "",
        kind: "cloud",
        authMode: "cookie",
      },
    ]);
  });

  it("preserves an existing locked Cloud org selection while reseeding", () => {
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_LOCK_TO_CLOUD__ = window.location.origin;
    writeStoredActiveBackend({ backendId: "locked-cloud", orgId: "org-1" });

    expect(readStoredBackends()[0]).toMatchObject({ id: "locked-cloud" });
    expect(readStoredActiveBackend()).toEqual({
      backendId: "locked-cloud",
      orgId: "org-1",
    });
  });

  it("does not seed a cookie-auth Cloud backend when locked cross-origin", () => {
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_LOCK_TO_CLOUD__ = "https://app.all-hands.dev";
    vi.stubEnv("VITE_SESSION_API_KEY", "");

    expect(readStoredBackends()).toEqual([]);
  });

  it("re-seeds the default Local backend when storage holds an empty array and launcher details are available", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
    vi.stubEnv("VITE_SESSION_API_KEY", "fresh-session-key");
    window.localStorage.setItem(BACKENDS_STORAGE_KEY, JSON.stringify([]));

    const result = readStoredBackends();

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({ id: "default-local", kind: "local" });
  });

  it("re-seeds the default Local backend when every stored entry is invalid and launcher details are available", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
    vi.stubEnv("VITE_SESSION_API_KEY", "fresh-session-key");
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([{ kind: "cloud" }, "not-an-object"]),
    );

    const result = readStoredBackends();

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({ id: "default-local", kind: "local" });
  });

  it("filters out backends with invalid shape", () => {
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([
        { id: "ok", name: "x", host: "y", apiKey: "z", kind: "local" },
        { id: "missing-kind", name: "x", host: "y", apiKey: "z" },
        { kind: "cloud" },
        "not-an-object",
      ]),
    );

    expect(readStoredBackends()).toEqual([
      { id: "ok", name: "x", host: "y", apiKey: "z", kind: "local" },
    ]);
  });

  it("preserves stored backends without API keys", () => {
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    const storedBackend: Backend = {
      id: "default-local",
      name: "Local",
      host: window.location.origin,
      apiKey: "",
      kind: "local",
    };
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([storedBackend]),
    );

    expect(readStoredBackends()).toEqual([storedBackend]);
  });

  it("syncs a stale default Local API key from env defaults", () => {
    vi.stubEnv("VITE_SESSION_API_KEY", "fresh-session-key");
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([
        {
          id: "default-local",
          name: "Local",
          host: window.location.origin,
          apiKey: "stored-session-key",
          kind: "local",
        },
      ]),
    );

    expect(readStoredBackends()[0]).toMatchObject({
      id: "default-local",
      apiKey: "fresh-session-key",
    });
  });

  it("syncs a stale default Local API key across localhost and 127.0.0.1", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    vi.stubEnv("VITE_SESSION_API_KEY", "fresh-session-key");
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([
        {
          id: "default-local",
          name: "Local",
          host: "http://localhost:8000",
          apiKey: "stored-session-key",
          kind: "local",
        },
      ]),
    );

    expect(readStoredBackends()[0]).toMatchObject({
      id: "default-local",
      host: "http://localhost:8000",
      apiKey: "fresh-session-key",
    });
  });

  it("preserves a custom backend API key instead of syncing from env defaults", () => {
    vi.stubEnv("VITE_SESSION_API_KEY", "fresh-session-key");
    const storedBackend: Backend = {
      id: "custom-local",
      name: "Custom Local",
      host: window.location.origin,
      apiKey: "stored-session-key",
      kind: "local",
    };
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([storedBackend]),
    );

    expect(readStoredBackends()[0]).toMatchObject({
      id: "custom-local",
      apiKey: "stored-session-key",
    });
  });

  it("preserves a user-edited non-loopback default Local backend API key", () => {
    vi.stubEnv("VITE_SESSION_API_KEY", "fresh-session-key");
    const storedBackend: Backend = {
      id: "default-local",
      name: "Edited Local",
      host: "https://example.com",
      apiKey: "stored-session-key",
      kind: "local",
    };
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([storedBackend]),
    );

    expect(readStoredBackends()[0]).toMatchObject({
      id: "default-local",
      host: "https://example.com",
      apiKey: "stored-session-key",
    });
  });

  it("round-trips active selection with orgId", () => {
    writeStoredActiveBackend({ backendId: "xyz", orgId: "org-1" });
    expect(readStoredActiveBackend()).toEqual({
      backendId: "xyz",
      orgId: "org-1",
    });
  });

  it("normalizes missing orgId to null", () => {
    writeStoredActiveBackend({ backendId: "xyz" });
    expect(readStoredActiveBackend()).toEqual({
      backendId: "xyz",
      orgId: null,
    });
  });

  it("prefers the tab-scoped active selection over the global fallback", () => {
    window.localStorage.setItem(
      ACTIVE_BACKEND_STORAGE_KEY,
      JSON.stringify({ backendId: "global-backend", orgId: null }),
    );
    window.sessionStorage.setItem(
      ACTIVE_BACKEND_STORAGE_KEY,
      JSON.stringify({ backendId: "tab-backend", orgId: "org-1" }),
    );

    expect(readStoredActiveBackend()).toEqual({
      backendId: "tab-backend",
      orgId: "org-1",
    });
  });

  it("falls back to the global active selection for new tabs", () => {
    window.localStorage.setItem(
      ACTIVE_BACKEND_STORAGE_KEY,
      JSON.stringify({ backendId: "global-backend", orgId: null }),
    );

    expect(readStoredActiveBackend()).toEqual({
      backendId: "global-backend",
      orgId: null,
    });
  });

  it("clears storage when active selection is set to null", () => {
    writeStoredActiveBackend({ backendId: "xyz", orgId: "o" });
    writeStoredActiveBackend(null);

    expect(
      window.sessionStorage.getItem(ACTIVE_BACKEND_STORAGE_KEY),
    ).toBeNull();
    expect(window.localStorage.getItem(ACTIVE_BACKEND_STORAGE_KEY)).toBeNull();
    expect(readStoredActiveBackend()).toBeNull();
  });

  it("returns null active selection when storage is malformed", () => {
    window.localStorage.setItem(ACTIVE_BACKEND_STORAGE_KEY, "{broken");
    expect(readStoredActiveBackend()).toBeNull();
  });
});

describe("backend-registry session key persistence", () => {
  const REMOTE_BACKEND: Backend = {
    id: "remote-agent",
    name: "Remote Agent Server",
    host: "https://agent.example.com",
    apiKey: "remote-session-key",
    kind: "local",
  };

  const CLOUD_TOKEN_BACKEND: Backend = {
    id: "cloud-token",
    name: "Cloud (API key)",
    host: "https://app.all-hands.dev",
    apiKey: "oauth-access-token",
    kind: "cloud",
    authMode: "api-key",
  };

  const LOOPBACK_BACKEND: Backend = {
    id: "local-agent",
    name: "Local Agent Server",
    host: "http://127.0.0.1:8000",
    apiKey: "loopback-session-key",
    kind: "local",
  };

  const MARS_BACKEND: Backend = {
    id: "mars-session",
    name: "MARS Session",
    host: "https://mars.example.com",
    apiKey: "mars-guest-key",
    kind: "local",
    marsSessionId: "session-1",
  };

  function readPersistedBackends(): Backend[] {
    const raw = window.localStorage.getItem(BACKENDS_STORAGE_KEY);
    return raw === null ? [] : (JSON.parse(raw) as Backend[]);
  }

  function readSessionKeyMap(): Record<string, string> {
    const raw = window.sessionStorage.getItem(SESSION_KEYS_STORAGE_KEY);
    return raw === null ? {} : (JSON.parse(raw) as Record<string, string>);
  }

  afterEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });

  it("persists only tab-scoped keys outside the record while keeping the in-memory view lossless", () => {
    writeStoredBackends([
      REMOTE_BACKEND,
      CLOUD_TOKEN_BACKEND,
      LOOPBACK_BACKEND,
    ]);

    expect(readPersistedBackends()).toEqual([
      { ...REMOTE_BACKEND, apiKey: "" },
      { ...CLOUD_TOKEN_BACKEND, apiKey: "" },
      LOOPBACK_BACKEND,
    ]);
    expect(readSessionKeyMap()).toEqual({
      "remote-agent": "remote-session-key",
      "cloud-token": "oauth-access-token",
    });
    expect(readStoredBackends()).toEqual([
      REMOTE_BACKEND,
      CLOUD_TOKEN_BACKEND,
      LOOPBACK_BACKEND,
    ]);
  });

  it("migrates a legacy plaintext record on read without losing the key or fields", () => {
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([{ ...REMOTE_BACKEND, connectionRevision: 3 }]),
    );

    expect(readStoredBackends()).toEqual([
      { ...REMOTE_BACKEND, connectionRevision: 3 },
    ]);
    expect(readPersistedBackends()).toEqual([
      { ...REMOTE_BACKEND, connectionRevision: 3, apiKey: "" },
    ]);
    expect(readSessionKeyMap()).toEqual({
      "remote-agent": "remote-session-key",
    });
  });

  it("surfaces a blank record when the tab-scoped key is gone", () => {
    writeStoredBackends([REMOTE_BACKEND]);
    window.sessionStorage.clear();

    expect(readStoredBackends()).toEqual([{ ...REMOTE_BACKEND, apiKey: "" }]);
  });

  it("drops the tab-scoped key when the backend is removed", () => {
    writeStoredBackends([REMOTE_BACKEND]);
    writeStoredBackends([LOOPBACK_BACKEND]);

    expect(readSessionKeyMap()).toEqual({});
  });

  it("keeps the public MARS guest key in the persisted record", () => {
    writeStoredBackends([MARS_BACKEND]);

    expect(readPersistedBackends()).toEqual([MARS_BACKEND]);
    expect(readSessionKeyMap()).toEqual({});
    expect(readStoredBackends()).toEqual([MARS_BACKEND]);
  });
});
