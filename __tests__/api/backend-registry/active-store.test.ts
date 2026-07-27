import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  getActiveBackend,
  getEffectiveLocalBackend,
  NO_BACKEND_ID,
  setActiveSelection,
  setRegisteredBackends,
  subscribeActiveBackend,
} from "#/api/backend-registry/active-store";
import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";
import type { Backend } from "#/api/backend-registry/types";

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  __resetActiveStoreForTests();
});

afterEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  vi.unstubAllEnvs();
  __resetActiveStoreForTests();
});

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:9000",
  apiKey: "k",
  kind: "local",
};

describe("active-store", () => {
  it("uses the no-backend sentinel when no backend details are available", () => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    vi.stubEnv("VITE_SESSION_API_KEY", "");
    __resetActiveStoreForTests();

    const { backend, orgId } = getActiveBackend();
    expect(backend.id).toBe(NO_BACKEND_ID);
    expect(orgId).toBeNull();
  });

  it("seeds the registry with a default local backend when host and API key are available", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
    vi.stubEnv("VITE_SESSION_API_KEY", "session-key");
    __resetActiveStoreForTests();

    const { backend, orgId } = getActiveBackend();
    expect(backend.id).toBe(SEEDED_DEFAULT_BACKEND_ID);
    expect(backend.kind).toBe("local");
    expect(orgId).toBeNull();
  });

  it("returns the registered backend matching the active selection", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });

    const { backend, orgId } = getActiveBackend();
    expect(backend).toEqual(cloudBackend);
    expect(orgId).toBe("org-2");
  });

  it("falls back to the first local backend when the active selection points at a removed entry", () => {
    setRegisteredBackends([cloudBackend, localBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: null });
    setRegisteredBackends([localBackend]);

    expect(getActiveBackend().backend).toEqual(localBackend);
    expect(getActiveBackend().orgId).toBeNull();
  });

  it("falls back to the first registered backend when the registry has no local entry", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection(null);

    expect(getActiveBackend().backend).toEqual(cloudBackend);
  });

  it("uses the active local backend as the effective local backend", () => {
    setRegisteredBackends([localBackend, cloudBackend]);
    setActiveSelection({ backendId: localBackend.id });

    expect(getEffectiveLocalBackend()).toEqual(localBackend);
  });

  it("does not borrow a registered local backend when the active backend is cloud", () => {
    setRegisteredBackends([localBackend, cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    expect(getEffectiveLocalBackend()).toBeNull();
  });

  it("notifies subscribers when selection changes", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeActiveBackend(listener);

    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    expect(listener).toHaveBeenCalled();

    unsubscribe();
    listener.mockClear();
    setActiveSelection(null);
    expect(listener).not.toHaveBeenCalled();
  });
});
