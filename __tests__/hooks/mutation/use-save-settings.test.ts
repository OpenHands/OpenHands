import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import React from "react";
import SettingsService from "#/api/settings-service/settings-service.api";

const trackMcpConfigUpdatedMock = vi.fn();
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackMcpConfigUpdated: trackMcpConfigUpdatedMock,
  }),
}));

const useSettingsMock = vi.fn();
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

import { useSaveSettings } from "#/hooks/mutation/use-save-settings";

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
};

describe("useSaveSettings - MCP tracking", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    // Default: no current mcp_config so any incoming config is treated as new.
    useSettingsMock.mockReturnValue({ data: {} });
  });

  it("calls trackMcpConfigUpdated with server counts when mcp_config changes", async () => {
    const { result } = renderHook(() => useSaveSettings(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      mcp_config: {
        sse_servers: [{ url: "http://sse1" }, { url: "http://sse2" }],
        stdio_servers: [{ name: "stdio1", command: "cmd", args: [] }],
        shttp_servers: [],
      },
    });

    await waitFor(() => {
      expect(trackMcpConfigUpdatedMock).toHaveBeenCalledWith({
        sseServersCount: 2,
        stdioServersCount: 1,
      });
    });
  });

  it("does not call trackMcpConfigUpdated when mcp_config is absent in the update", async () => {
    const { result } = renderHook(() => useSaveSettings(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({ user_consents_to_analytics: true });

    expect(trackMcpConfigUpdatedMock).not.toHaveBeenCalled();
  });

  it("does not call trackMcpConfigUpdated when mcp_config reference is unchanged", async () => {
    const sharedConfig = {
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    };
    useSettingsMock.mockReturnValue({
      data: { mcp_config: sharedConfig },
    });

    const { result } = renderHook(() => useSaveSettings(), {
      wrapper: createWrapper(),
    });

    // Passing the same object reference — should not trigger tracking.
    await result.current.mutateAsync({ mcp_config: sharedConfig });

    expect(trackMcpConfigUpdatedMock).not.toHaveBeenCalled();
  });

  it("counts zero servers correctly when server arrays are empty", async () => {
    const { result } = renderHook(() => useSaveSettings(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      mcp_config: { sse_servers: [], stdio_servers: [], shttp_servers: [] },
    });

    await waitFor(() => {
      expect(trackMcpConfigUpdatedMock).toHaveBeenCalledWith({
        sseServersCount: 0,
        stdioServersCount: 0,
      });
    });
  });
});
