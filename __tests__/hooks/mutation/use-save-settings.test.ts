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
        sse1: { transport: "sse", url: "http://sse1" },
        sse2: { transport: "sse", url: "http://sse2" },
        stdio1: { transport: "stdio", command: "cmd", args: [] },
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
    const sharedConfig = {};
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
      mcp_config: {},
    });

    await waitFor(() => {
      expect(trackMcpConfigUpdatedMock).toHaveBeenCalledWith({
        sseServersCount: 0,
        stdioServersCount: 0,
      });
    });
  });
});

describe("useSaveSettings - LLM api_key handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useSettingsMock.mockReturnValue({ data: {} });
  });

  it("drops an empty api_key from the diff so the stored key is preserved", async () => {
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const { result } = renderHook(() => useSaveSettings(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      agent_settings_diff: {
        llm: { model: "anthropic/claude-sonnet-4-5", api_key: "" },
      },
    });

    expect(saveSpy).toHaveBeenCalledTimes(1);
    const sent = saveSpy.mock.calls[0][0] as Record<string, unknown>;
    const llm = (sent.agent_settings_diff as Record<string, unknown>)
      .llm as Record<string, unknown>;
    expect(llm.model).toBe("anthropic/claude-sonnet-4-5");
    expect(llm.api_key).toBeUndefined();
  });

  it("forwards a non-empty api_key (trimmed)", async () => {
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const { result } = renderHook(() => useSaveSettings(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      agent_settings_diff: {
        llm: { api_key: "  sk-real-key  " },
      },
    });

    const llm = (
      (saveSpy.mock.calls[0][0] as Record<string, unknown>)
        .agent_settings_diff as Record<string, unknown>
    ).llm as Record<string, unknown>;
    expect(llm.api_key).toBe("sk-real-key");
  });

  it("does not touch api_key when it is absent from the diff", async () => {
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const { result } = renderHook(() => useSaveSettings(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      agent_settings_diff: { llm: { model: "openai/gpt-4o" } },
    });

    const llm = (
      (saveSpy.mock.calls[0][0] as Record<string, unknown>)
        .agent_settings_diff as Record<string, unknown>
    ).llm as Record<string, unknown>;
    expect(llm.api_key).toBeUndefined();
  });
});
