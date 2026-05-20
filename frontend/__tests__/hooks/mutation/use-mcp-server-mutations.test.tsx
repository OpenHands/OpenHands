import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useAddMcpServer } from "#/hooks/mutation/use-add-mcp-server";
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

// Mock SettingsService
vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: {
    getSettings: vi.fn(),
    saveSettings: vi.fn(),
  },
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("MCP Server Mutation Hooks", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useSelectedOrganizationStore.setState({ organizationId: "test-org-id" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("useAddMcpServer", () => {
    it("should fetch fresh settings at mutation time, not use stale data", async () => {
      // This tests the fix for the stale closure bug
      // The hook should call getSettings() inside mutationFn, not rely on
      // settings captured at render time

      const getSettingsSpy = vi.mocked(SettingsService.getSettings);
      const saveSettingsSpy = vi.mocked(SettingsService.saveSettings);

      // First call returns initial state with 1 server
      getSettingsSpy.mockResolvedValue({
        agent_settings: {
          mcp_config: {
            mcpServers: {
              existing: { url: "https://existing.com", transport: "sse" },
            },
          },
        },
      } as any);

      saveSettingsSpy.mockResolvedValue(true);

      const { result } = renderHook(() => useAddMcpServer(), {
        wrapper: createWrapper(),
      });

      // Add a new server
      result.current.mutate({
        type: "sse",
        url: "https://new-server.com",
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Verify getSettings was called during mutation (fresh fetch)
      expect(getSettingsSpy).toHaveBeenCalledTimes(1);

      // Verify saveSettings was called with both servers
      expect(saveSettingsSpy).toHaveBeenCalledWith({
        agent_settings_diff: {
          mcp_config: {
            mcpServers: expect.objectContaining({
              existing: expect.objectContaining({ url: "https://existing.com" }),
            }),
          },
        },
      });
    });

    it("should handle adding server when no existing config", async () => {
      const getSettingsSpy = vi.mocked(SettingsService.getSettings);
      const saveSettingsSpy = vi.mocked(SettingsService.saveSettings);

      // Return settings with no mcp_config
      getSettingsSpy.mockResolvedValue({
        agent_settings: {},
      } as any);

      saveSettingsSpy.mockResolvedValue(true);

      const { result } = renderHook(() => useAddMcpServer(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({
        type: "sse",
        url: "https://first-server.com",
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(saveSettingsSpy).toHaveBeenCalledWith({
        agent_settings_diff: {
          mcp_config: {
            mcpServers: {
              sse: {
                url: "https://first-server.com",
                transport: "sse",
              },
            },
          },
        },
      });
    });

    it("should return early if getSettings returns null", async () => {
      const getSettingsSpy = vi.mocked(SettingsService.getSettings);
      const saveSettingsSpy = vi.mocked(SettingsService.saveSettings);

      getSettingsSpy.mockResolvedValue(null as any);

      const { result } = renderHook(() => useAddMcpServer(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({
        type: "sse",
        url: "https://server.com",
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // saveSettings should NOT be called
      expect(saveSettingsSpy).not.toHaveBeenCalled();
    });
  });

  describe("useDeleteMcpServer", () => {
    it("should fetch fresh settings and delete the correct server", async () => {
      const getSettingsSpy = vi.mocked(SettingsService.getSettings);
      const saveSettingsSpy = vi.mocked(SettingsService.saveSettings);

      // Settings with 3 servers
      getSettingsSpy.mockResolvedValue({
        agent_settings: {
          mcp_config: {
            mcpServers: {
              server1: { url: "https://server1.com", transport: "sse" },
              server2: { url: "https://server2.com", transport: "sse" },
              server3: { url: "https://server3.com", transport: "sse" },
            },
          },
        },
      } as any);

      saveSettingsSpy.mockResolvedValue(true);

      const { result } = renderHook(() => useDeleteMcpServer(), {
        wrapper: createWrapper(),
      });

      // Delete server2 (index 1 in the sse_servers array)
      result.current.mutate("sse:1");

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Verify fresh settings were fetched
      expect(getSettingsSpy).toHaveBeenCalledTimes(1);

      // Verify saveSettings was called with only 2 servers
      const savedPayload = saveSettingsSpy.mock.calls[0][0] as {
        agent_settings_diff: { mcp_config: { mcpServers: Record<string, unknown> } };
      };
      const savedConfig = savedPayload.agent_settings_diff.mcp_config;
      const serverNames = Object.keys(savedConfig.mcpServers);
      expect(serverNames).toHaveLength(2);
    });

    it("should handle deleting from empty config gracefully", async () => {
      const getSettingsSpy = vi.mocked(SettingsService.getSettings);
      const saveSettingsSpy = vi.mocked(SettingsService.saveSettings);

      getSettingsSpy.mockResolvedValue({
        agent_settings: {
          mcp_config: null,
        },
      } as any);

      saveSettingsSpy.mockResolvedValue(true);

      const { result } = renderHook(() => useDeleteMcpServer(), {
        wrapper: createWrapper(),
      });

      // Try to delete from empty config
      result.current.mutate("sse:0");

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Should still call saveSettings with empty/null config
      expect(saveSettingsSpy).toHaveBeenCalled();
    });
  });

  describe("useUpdateMcpServer", () => {
    it("should fetch fresh settings and update the correct server", async () => {
      const getSettingsSpy = vi.mocked(SettingsService.getSettings);
      const saveSettingsSpy = vi.mocked(SettingsService.saveSettings);

      getSettingsSpy.mockResolvedValue({
        agent_settings: {
          mcp_config: {
            mcpServers: {
              myserver: { url: "https://old-url.com", transport: "sse" },
            },
          },
        },
      } as any);

      saveSettingsSpy.mockResolvedValue(true);

      const { result } = renderHook(() => useUpdateMcpServer(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({
        serverId: "sse:0",
        server: {
          type: "sse",
          url: "https://new-url.com",
        },
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Verify fresh settings were fetched
      expect(getSettingsSpy).toHaveBeenCalledTimes(1);

      // Verify the URL was updated
      const savedPayload = saveSettingsSpy.mock.calls[0][0] as {
        agent_settings_diff: {
          mcp_config: { mcpServers: Record<string, { url: string }> };
        };
      };
      const savedConfig = savedPayload.agent_settings_diff.mcp_config;
      const serverUrls = Object.values(savedConfig.mcpServers).map(
        (s) => s.url,
      );
      expect(serverUrls).toContain("https://new-url.com");
      expect(serverUrls).not.toContain("https://old-url.com");
    });
  });

  describe("Concurrent mutation handling", () => {
    it("should use fresh settings for each mutation in rapid succession", async () => {
      const getSettingsSpy = vi.mocked(SettingsService.getSettings);
      const saveSettingsSpy = vi.mocked(SettingsService.saveSettings);

      // Simulate settings changing between calls
      let callCount = 0;
      getSettingsSpy.mockImplementation(async () => {
        callCount++;
        return {
          agent_settings: {
            mcp_config: {
              mcpServers:
                callCount === 1
                  ? { server1: { url: "https://s1.com", transport: "sse" } }
                  : {
                      server1: { url: "https://s1.com", transport: "sse" },
                      server2: { url: "https://s2.com", transport: "sse" },
                    },
            },
          },
        } as any;
      });

      saveSettingsSpy.mockResolvedValue(true);

      const { result } = renderHook(() => useAddMcpServer(), {
        wrapper: createWrapper(),
      });

      // First mutation
      result.current.mutate({ type: "sse", url: "https://new1.com" });

      await waitFor(() => {
        expect(getSettingsSpy).toHaveBeenCalledTimes(1);
      });

      // Second mutation (settings have changed)
      result.current.mutate({ type: "sse", url: "https://new2.com" });

      await waitFor(() => {
        expect(getSettingsSpy).toHaveBeenCalledTimes(2);
      });

      // Both mutations fetched fresh settings
      expect(getSettingsSpy).toHaveBeenCalledTimes(2);
    });
  });

  describe("Error handling", () => {
    it("should handle getSettings failure gracefully", async () => {
      const getSettingsSpy = vi.mocked(SettingsService.getSettings);

      getSettingsSpy.mockRejectedValue(new Error("Network error"));

      const { result } = renderHook(() => useAddMcpServer(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ type: "sse", url: "https://server.com" });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toBeDefined();
    });

    it("should handle saveSettings failure", async () => {
      const getSettingsSpy = vi.mocked(SettingsService.getSettings);
      const saveSettingsSpy = vi.mocked(SettingsService.saveSettings);

      getSettingsSpy.mockResolvedValue({
        agent_settings: { mcp_config: null },
      } as any);

      saveSettingsSpy.mockRejectedValue(new Error("Save failed"));

      const { result } = renderHook(() => useAddMcpServer(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ type: "sse", url: "https://server.com" });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toBeDefined();
    });
  });
});
