import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useAddMcpServer } from "#/hooks/mutation/use-add-mcp-server";
import { SUPER_ADMIN_SETUP_QUERY_KEYS } from "#/hooks/query/query-keys";

const useSettingsMock = vi.fn();
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

const createWrapper = () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client }, children);
  };
};

describe("useAddMcpServer", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useSettingsMock.mockReturnValue({
      data: {
        mcp_config: {
          github: {
            transport: "http",
            url: "https://github.example/mcp",
            auth: { strategy: "bearer", value: "**********" },
          },
        },
      },
    });
  });

  it("adds only one named entry and leaves siblings out of the mutation", async () => {
    const createSpy = vi
      .spyOn(SettingsService, "createMcpServer")
      .mockResolvedValue(true);
    const fetchEncryptedSpy = vi.spyOn(SettingsService, "fetchSettingsFromApi");
    const { result } = renderHook(() => useAddMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      id: "",
      type: "shttp",
      name: "docs",
      url: "https://docs.example/mcp",
    });

    expect(createSpy).toHaveBeenCalledOnce();
    expect(createSpy).toHaveBeenCalledWith("docs", {
      transport: "http",
      url: "https://docs.example/mcp",
    });
    expect(fetchEncryptedSpy).not.toHaveBeenCalled();
  });

  it("allocates a deterministic suffix without overwriting an existing key", async () => {
    useSettingsMock.mockReturnValue({
      data: {
        mcp_config: {
          slack: { transport: "stdio", command: "npx" },
        },
      },
    });
    const createSpy = vi
      .spyOn(SettingsService, "createMcpServer")
      .mockResolvedValue(true);
    const { result } = renderHook(() => useAddMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      id: "",
      type: "stdio",
      name: "slack",
      command: "npx",
    });

    expect(createSpy).toHaveBeenCalledWith("slack_1", {
      transport: "stdio",
      command: "npx",
    });
  });

  it("marks the Super Admin setup guide's progress stale after adding a server", async () => {
    vi.spyOn(SettingsService, "createMcpServer").mockResolvedValue(true);
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    const setupStateKey = SUPER_ADMIN_SETUP_QUERY_KEYS.state("cloud-1", 0);
    client.setQueryData(setupStateKey, { guide_steps: { mcp_server: false } });
    const { result } = renderHook(() => useAddMcpServer(), {
      wrapper: ({ children }: { children: React.ReactNode }) =>
        React.createElement(QueryClientProvider, { client }, children),
    });

    await result.current.mutateAsync({
      id: "",
      type: "shttp",
      name: "docs",
      url: "https://docs.example/mcp",
    });

    expect(client.getQueryState(setupStateKey)?.isInvalidated).toBe(true);
  });

  it("fails instead of reporting a successful install before settings load", async () => {
    useSettingsMock.mockReturnValue({ data: undefined });
    const createSpy = vi.spyOn(SettingsService, "createMcpServer");
    const { result } = renderHook(() => useAddMcpServer(), {
      wrapper: createWrapper(),
    });

    await expect(
      result.current.mutateAsync({
        id: "",
        type: "stdio",
        name: "filesystem",
        command: "npx",
      }),
    ).rejects.toThrow("MCP settings are still loading");
    expect(createSpy).not.toHaveBeenCalled();
  });
});
