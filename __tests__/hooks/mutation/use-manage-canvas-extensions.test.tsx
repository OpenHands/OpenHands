import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  useSetCanvasExtensionEnabled,
  useUninstallCanvasExtension,
} from "#/hooks/mutation/use-manage-canvas-extensions";
import { getPinnedHomeRouteKey } from "#/hooks/use-pinned-home-route";

const backend: Backend = {
  id: "test-backend",
  name: "Test backend",
  host: "http://127.0.0.1:8000",
  apiKey: "test-key",
  kind: "local",
};

const pinKey = getPinnedHomeRouteKey(backend.id, null);

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
}

describe("useManageCanvasExtensions pin lifecycle", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setRegisteredBackends([backend]);
    setActiveSelection({ backendId: backend.id });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    setActiveSelection(null);
    setRegisteredBackends([]);
  });

  it("clears the pinned route when the extension is disabled", async () => {
    window.localStorage.setItem(
      pinKey,
      JSON.stringify("/extensions/demo-extension/dashboard"),
    );
    vi.spyOn(CanvasExtensionsService, "setEnabled").mockResolvedValue({
      name: "demo-extension",
      enabled: false,
    });

    const { result } = renderHook(() => useSetCanvasExtensionEnabled(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ name: "demo-extension", enabled: false });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(window.localStorage.getItem(pinKey)).toBeNull();
  });

  it("does not clear the pinned route when the extension is enabled", async () => {
    window.localStorage.setItem(
      pinKey,
      JSON.stringify("/extensions/demo-extension/dashboard"),
    );
    vi.spyOn(CanvasExtensionsService, "setEnabled").mockResolvedValue({
      name: "demo-extension",
      enabled: true,
    });

    const { result } = renderHook(() => useSetCanvasExtensionEnabled(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ name: "demo-extension", enabled: true });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(window.localStorage.getItem(pinKey)).toBe(
      JSON.stringify("/extensions/demo-extension/dashboard"),
    );
  });

  it("clears the pinned route when the extension is uninstalled", async () => {
    window.localStorage.setItem(
      pinKey,
      JSON.stringify("/extensions/demo-extension/dashboard"),
    );
    vi.spyOn(CanvasExtensionsService, "uninstall").mockResolvedValue({
      message: "uninstalled",
    });

    const { result } = renderHook(() => useUninstallCanvasExtension(), {
      wrapper: createWrapper(),
    });

    result.current.mutate("demo-extension");

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(window.localStorage.getItem(pinKey)).toBeNull();
  });
});
