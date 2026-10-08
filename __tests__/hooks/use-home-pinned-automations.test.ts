import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SettingsService from "#/api/settings-service/settings-service.api";
import { DEFAULT_SETTINGS } from "#/services/settings";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  applyPinnedOrder,
  getHomePinnedAutomationsKey,
  HOME_PINNED_AUTOMATIONS_KEY,
  movePinnedId,
  useHomePinnedAutomations,
} from "#/hooks/use-home-pinned-automations";

vi.mock("#/api/settings-service/settings-service.api");

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

function renderPinnedHook<T>(hook: () => T) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(
      QueryClientProvider,
      { client: queryClient },
      React.createElement(ActiveBackendProvider, null, children),
    );
  return renderHook(hook, { wrapper });
}

function activate(backend: Backend) {
  setRegisteredBackends([backend]);
  setActiveSelection({ backendId: backend.id });
}

/** Pins the mocked agent-server holds; saves replace it, like the server. */
let serverPins: string[] | undefined;

beforeEach(() => {
  window.localStorage.clear();
  serverPins = undefined;
  vi.mocked(SettingsService.getSettings).mockImplementation(async () => ({
    ...DEFAULT_SETTINGS,
    home_pinned_automations: serverPins,
  }));
  vi.mocked(SettingsService.saveSettings).mockImplementation(async (update) => {
    serverPins = update.home_pinned_automations;
    return true;
  });
});

afterEach(() => {
  __resetActiveStoreForTests();
  vi.clearAllMocks();
});

describe("useHomePinnedAutomations on a local backend", () => {
  it("reads pins from server settings and saves each change as the full ordered list", async () => {
    // Arrange
    activate(localBackend);
    serverPins = ["a", "b"];
    const { result } = renderPinnedHook(() => useHomePinnedAutomations());
    await waitFor(() => expect(result.current.pinnedIds).toEqual(["a", "b"]));

    // Act + Assert: each change shows in the list and is saved in full.
    act(() => result.current.pin("c"));
    await waitFor(() =>
      expect(result.current.pinnedIds).toEqual(["a", "b", "c"]),
    );
    act(() => result.current.reorder("c", "a", "before"));
    await waitFor(() =>
      expect(result.current.pinnedIds).toEqual(["c", "a", "b"]),
    );
    act(() => result.current.unpin("a"));
    await waitFor(() => expect(result.current.pinnedIds).toEqual(["c", "b"]));

    await waitFor(() =>
      expect(
        vi
          .mocked(SettingsService.saveSettings)
          .mock.calls.map(([update]) => update),
      ).toEqual([
        { home_pinned_automations: ["a", "b", "c"] },
        { home_pinned_automations: ["c", "a", "b"] },
        { home_pinned_automations: ["c", "b"] },
      ]),
    );
  });

  it("moves legacy localStorage pins to the server once, then removes the local copy", async () => {
    // Arrange: the dashboard and the activity list each mount the hook.
    activate(localBackend);
    const legacyKey = getHomePinnedAutomationsKey(localBackend.id, null);
    window.localStorage.setItem(legacyKey, JSON.stringify(["legacy", "a"]));
    serverPins = ["a"];

    // Act
    const { result } = renderPinnedHook(() => [
      useHomePinnedAutomations(),
      useHomePinnedAutomations(),
    ]);

    // Assert
    await waitFor(() =>
      expect(window.localStorage.getItem(legacyKey)).toBe(null),
    );
    expect(SettingsService.saveSettings).toHaveBeenCalledExactlyOnceWith({
      home_pinned_automations: ["a", "legacy"],
    });
    expect(result.current[0].pinnedIds).toEqual(["a", "legacy"]);
  });

  it("keeps the legacy localStorage pins when the server save fails", async () => {
    // Arrange
    activate(localBackend);
    const legacyKey = getHomePinnedAutomationsKey(localBackend.id, null);
    window.localStorage.setItem(legacyKey, JSON.stringify(["legacy"]));
    vi.mocked(SettingsService.saveSettings).mockRejectedValue(
      new Error("offline"),
    );

    // Act
    renderPinnedHook(() => useHomePinnedAutomations());

    // Assert
    await waitFor(() =>
      expect(SettingsService.saveSettings).toHaveBeenCalledOnce(),
    );
    expect(window.localStorage.getItem(legacyKey)).toBe(
      JSON.stringify(["legacy"]),
    );
  });
});

describe("useHomePinnedAutomations on a cloud backend", () => {
  it("keeps pins in localStorage and never writes them to settings", async () => {
    // Arrange
    activate(cloudBackend);
    const { result } = renderPinnedHook(() => useHomePinnedAutomations());

    // Act
    act(() => result.current.pin("a"));

    // Assert
    await waitFor(() => expect(result.current.pinnedIds).toEqual(["a"]));
    expect(
      window.localStorage.getItem(
        getHomePinnedAutomationsKey(cloudBackend.id, null),
      ),
    ).toBe(JSON.stringify(["a"]));
    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
  });
});

describe("getHomePinnedAutomationsKey", () => {
  it("scopes the storage key by backend and org", () => {
    expect(getHomePinnedAutomationsKey("backend-a", "org-1")).toBe(
      `${HOME_PINNED_AUTOMATIONS_KEY}:backend-a:org-1`,
    );
    expect(getHomePinnedAutomationsKey("backend-a", null)).toBe(
      `${HOME_PINNED_AUTOMATIONS_KEY}:backend-a:-`,
    );
    expect(getHomePinnedAutomationsKey("backend-a", "org-1")).not.toBe(
      getHomePinnedAutomationsKey("backend-b", "org-1"),
    );
  });
});

describe("movePinnedId", () => {
  it("reorders an id before or after a target", () => {
    expect(movePinnedId(["a", "b", "c"], "c", "a", "before")).toEqual([
      "c",
      "a",
      "b",
    ]);
    expect(movePinnedId(["a", "b", "c"], "a", "b", "after")).toEqual([
      "b",
      "a",
      "c",
    ]);
  });
});

describe("applyPinnedOrder", () => {
  it("applies a preferred order while keeping unknown base ids", () => {
    expect(applyPinnedOrder(["a", "b", "c"], ["c", "a"])).toEqual([
      "c",
      "a",
      "b",
    ]);
  });
});
