import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  BACKEND_HEALTH_STORAGE_KEY,
  MAX_CONSECUTIVE_FAILURES,
} from "#/api/backend-registry/health-storage";
import {
  __resetHealthStoreForTests,
  getBackendHealthEntry,
  recordBackendFailure,
  recordBackendSuccess,
  resetBackendHealth,
  subscribeBackendHealth,
} from "#/api/backend-registry/health-store";

const BACKEND_ID = "backend-under-test";

beforeEach(() => {
  window.localStorage.clear();
  __resetHealthStoreForTests();
});

afterEach(() => {
  window.localStorage.clear();
  __resetHealthStoreForTests();
});

describe("backend health store", () => {
  it("increments the failure count and persists to localStorage without disabling polling below the cap", () => {
    // Arrange / Act
    recordBackendFailure(BACKEND_ID, new Error("ECONNREFUSED"));
    recordBackendFailure(BACKEND_ID, new Error("ECONNREFUSED"));

    // Assert
    const entry = getBackendHealthEntry(BACKEND_ID);
    expect(entry).toMatchObject({
      consecutiveFailures: 2,
      disabled: false,
      lastError: "ECONNREFUSED",
    });

    const persisted = JSON.parse(
      window.localStorage.getItem(BACKEND_HEALTH_STORAGE_KEY) ?? "{}",
    );
    expect(persisted[BACKEND_ID]).toMatchObject({
      consecutiveFailures: 2,
      disabled: false,
    });
  });

  it("flips disabled=true once consecutive failures hit the cap and persists that flag", () => {
    // Arrange / Act
    for (let i = 0; i < MAX_CONSECUTIVE_FAILURES; i += 1) {
      recordBackendFailure(BACKEND_ID, new Error("timeout"));
    }

    // Assert — the cap is reached; polling-consumers will see disabled.
    const entry = getBackendHealthEntry(BACKEND_ID);
    expect(entry?.consecutiveFailures).toBe(MAX_CONSECUTIVE_FAILURES);
    expect(entry?.disabled).toBe(true);

    const persisted = JSON.parse(
      window.localStorage.getItem(BACKEND_HEALTH_STORAGE_KEY) ?? "{}",
    );
    expect(persisted[BACKEND_ID].disabled).toBe(true);
  });

  it("caps the failure count at the max when a disabled backend fails another recheck", () => {
    for (let i = 0; i < MAX_CONSECUTIVE_FAILURES; i += 1) {
      recordBackendFailure(BACKEND_ID, new Error("timeout"));
    }

    recordBackendFailure(BACKEND_ID, new Error("still down"));

    const entry = getBackendHealthEntry(BACKEND_ID);
    expect(entry).toMatchObject({
      consecutiveFailures: MAX_CONSECUTIVE_FAILURES,
      disabled: true,
      lastError: "still down",
    });

    const persisted = JSON.parse(
      window.localStorage.getItem(BACKEND_HEALTH_STORAGE_KEY) ?? "{}",
    );
    expect(persisted[BACKEND_ID].consecutiveFailures).toBe(
      MAX_CONSECUTIVE_FAILURES,
    );
  });

  it("clears the entry (and storage) and notifies subscribers when the backend recovers or the user edits its config", () => {
    // Arrange — record one failure so there is something to clear, and
    // subscribe so we can confirm listeners get notified.
    recordBackendFailure(BACKEND_ID, new Error("boom"));
    let notifications = 0;
    const unsubscribe = subscribeBackendHealth(() => {
      notifications += 1;
    });

    // Act — `recordBackendSuccess` (probe recovers) and
    // `resetBackendHealth` (user edits host/apiKey) share the same
    // clear-entry semantics, so we cover both in one go.
    recordBackendSuccess(BACKEND_ID);
    recordBackendFailure(BACKEND_ID, new Error("again"));
    resetBackendHealth(BACKEND_ID);
    unsubscribe();

    // Assert
    expect(getBackendHealthEntry(BACKEND_ID)).toBeNull();
    expect(window.localStorage.getItem(BACKEND_HEALTH_STORAGE_KEY)).toBeNull();
    expect(notifications).toBeGreaterThanOrEqual(3);
  });
});
