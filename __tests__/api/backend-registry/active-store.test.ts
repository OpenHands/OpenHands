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
import { MAX_CONSECUTIVE_FAILURES } from "#/api/backend-registry/health-storage";
import {
  __resetHealthStoreForTests,
  recordBackendFailure,
} from "#/api/backend-registry/health-store";
import type { Backend } from "#/api/backend-registry/types";

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  __resetHealthStoreForTests();
  __resetActiveStoreForTests();
});

afterEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  vi.unstubAllEnvs();
  __resetHealthStoreForTests();
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

const secondLocalBackend: Backend = {
  id: "local-2",
  name: "Local 2",
  host: "http://localhost:9001",
  apiKey: "k2",
  kind: "local",
};

function markBackendUnhealthy(id: string): void {
  for (let i = 0; i < MAX_CONSECUTIVE_FAILURES; i += 1) {
    recordBackendFailure(id, new Error("connection failed"));
  }
}

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

  it("falls back to a healthy local backend when a cloud backend is registered first", () => {
    setRegisteredBackends([cloudBackend, localBackend]);
    setActiveSelection(null);

    expect(getActiveBackend().backend).toEqual(localBackend);
  });

  it("skips an unhealthy local backend in favor of a healthy one further down", () => {
    markBackendUnhealthy(localBackend.id);
    setRegisteredBackends([localBackend, secondLocalBackend]);
    setActiveSelection(null);

    expect(getActiveBackend().backend).toEqual(secondLocalBackend);
  });

  it("still selects a local backend when every local backend is unhealthy", () => {
    markBackendUnhealthy(localBackend.id);
    markBackendUnhealthy(secondLocalBackend.id);
    setRegisteredBackends([cloudBackend, localBackend, secondLocalBackend]);
    setActiveSelection(null);

    const { backend } = getActiveBackend();
    expect(backend.kind).toBe("local");
    expect(backend.id).not.toBe(NO_BACKEND_ID);
    // Deterministic: first local backend in insertion order.
    expect(backend).toEqual(localBackend);
  });

  it("selects a single healthy local backend at the first registry position", () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection(null);

    expect(getActiveBackend().backend).toEqual(localBackend);
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

  it("keeps an explicit cloud selection even when a healthy local backend exists", () => {
    setRegisteredBackends([localBackend, cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });

    const { backend, orgId } = getActiveBackend();
    expect(backend).toEqual(cloudBackend);
    expect(orgId).toBe("org-2");
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
