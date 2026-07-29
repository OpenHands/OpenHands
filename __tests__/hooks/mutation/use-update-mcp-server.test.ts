import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import React from "react";
import SettingsService, {
  type SettingsApiResponse,
} from "#/api/settings-service/settings-service.api";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";

const useSettingsMock = vi.fn();
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import {
  __resetMcpHealthStoreForTests,
  getMcpHealthSnapshot,
  setMcpServerHealth,
} from "#/api/mcp-health/mcp-health-store";
import type { MCPServerConfig } from "#/types/mcp-server";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

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

describe("useUpdateMcpServer - stdio credential preservation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
  });

  it("saves the encrypted stdio env, not the redacted placeholder, when a stdio server is renamed", async () => {
    // The redacted settings the editor reads from still carry the original name
    // ("old_name") and redacted env. The user renames to "new_name" and leaves
    // the secret env value as the redaction placeholder.
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: {
          mcp_config: {
            old_name: {
              command: "npx",
              env: { API_KEY: REDACTED_MCP_SECRET_VALUE },
            },
          },
        },
      },
    });

    vi.spyOn(SettingsService, "fetchSettingsFromApi").mockResolvedValue({
      agent_settings: {
        mcp_config: {
          old_name: {
            command: "npx",
            env: { API_KEY: "gAAAAA-encrypted-api-key" },
          },
        },
      },
    } as unknown as SettingsApiResponse);

    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      serverId: "stdio-0",
      server: {
        id: "stdio-0",
        type: "stdio",
        name: "new_name",
        command: "npx",
        env: { API_KEY: REDACTED_MCP_SECRET_VALUE },
      },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(SettingsService.saveSettings).toHaveBeenCalledTimes(1);
    const savedDiff = vi.mocked(SettingsService.saveSettings).mock.calls[0][0]
      .agent_settings_diff as Record<string, unknown> | undefined;
    const savedSdkConfig = savedDiff?.mcp_config;
    expect(savedSdkConfig).toMatchObject({
      new_name: {
        command: "npx",
        env: { API_KEY: "gAAAAA-encrypted-api-key" },
      },
    });
    // The literal placeholder must never round-trip into the saved config.
    expect(JSON.stringify(savedSdkConfig)).not.toContain(
      REDACTED_MCP_SECRET_VALUE,
    );
  });
});

describe("useUpdateMcpServer - health reset", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    __resetMcpHealthStoreForTests();
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
  });

  it("clears the server's stored health verdict when its config is saved", async () => {
    // A verdict from before the edit describes a different credential/config
    // and must not survive the save as a stale (possibly false) healthy.
    const server: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      name: "custom",
      url: "https://mcp.example.com/mcp",
    };
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: {
          mcp_config: { custom: { url: server.url, transport: "http" } },
        },
      },
    });
    const key = getMcpServerHealthKey(server);
    setMcpServerHealth(key, {
      status: "healthy",
      verification: "verified",
      toolCount: 1,
      checkedAt: 1,
    });

    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync({ serverId: server.id, server });

    await waitFor(() =>
      expect(getMcpHealthSnapshot()[key]).toBeUndefined(),
    );
  });
});
