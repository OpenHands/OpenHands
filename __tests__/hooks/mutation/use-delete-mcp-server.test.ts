import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import React from "react";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  __resetMcpHealthStoreForTests,
  getMcpHealthSnapshot,
  setMcpServerHealth,
} from "#/api/mcp-health/mcp-health-store";
import type { MCPServerConfig } from "#/types/mcp-server";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

const useSettingsMock = vi.fn();
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(
      QueryClientProvider,
      { client: queryClient },
      children,
    );
  }
  return Wrapper;
};

describe("useDeleteMcpServer - health reset", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    __resetMcpHealthStoreForTests();
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
  });

  it("drops the deleted server's health verdict", async () => {
    // A later server can legitimately produce the same health key (e.g. a
    // reinstall of the same catalog entry); it must start unchecked instead
    // of inheriting the deleted server's verdict.
    const target: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      name: "custom",
      url: "https://mcp.example.com/mcp",
    };
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: {
          mcp_config: { custom: { url: target.url, transport: "http" } },
        },
      },
    });
    const key = getMcpServerHealthKey(target);
    setMcpServerHealth(key, {
      status: "healthy",
      verification: "verified",
      toolCount: 1,
      checkedAt: 1,
    });

    const { result } = renderHook(() => useDeleteMcpServer(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync(target);

    await waitFor(() => expect(getMcpHealthSnapshot()[key]).toBeUndefined());
  });
});
