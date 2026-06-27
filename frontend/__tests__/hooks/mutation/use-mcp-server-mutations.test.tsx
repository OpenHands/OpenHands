import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useAddMcpServer } from "#/hooks/mutation/use-add-mcp-server";
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

describe("MCP Server Mutation Hooks", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useSelectedOrganizationStore.setState({ organizationId: "test-org-id" });
  });

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

  describe("useAddMcpServer", () => {
    it("fetches fresh settings at mutation time", async () => {
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue({
          agent_settings: {
            mcp_config: {
              mcpServers: {
                existing: { url: "https://existing.com", transport: "sse" },
              },
            },
          },
        } as any);

      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      const { result } = renderHook(() => useAddMcpServer(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({
        type: "sse",
        url: "https://new-server.com",
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(getSettingsSpy).toHaveBeenCalledTimes(1);
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

    it("handles adding server when no existing config", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        agent_settings: {},
      } as any);

      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

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

    it("proceeds with empty config when getSettings returns null", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue(null as any);

      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

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

      // Implementation handles null gracefully and proceeds
      expect(saveSettingsSpy).toHaveBeenCalledWith({
        agent_settings_diff: {
          mcp_config: {
            mcpServers: {
              sse: {
                url: "https://server.com",
                transport: "sse",
              },
            },
          },
        },
      });
    });
  });

  describe("useDeleteMcpServer", () => {
    it("deletes the correct server by index", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
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

      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      const { result } = renderHook(() => useDeleteMcpServer(), {
        wrapper: createWrapper(),
      });

      // Use hyphen separator as per implementation: serverId.split("-")
      result.current.mutate("sse-1");

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      const savedPayload = saveSettingsSpy.mock.calls[0][0] as {
        agent_settings_diff: {
          mcp_config: { mcpServers: Record<string, unknown> } | null;
        };
      };
      const savedConfig = savedPayload.agent_settings_diff.mcp_config;
      const serverNames = Object.keys(savedConfig?.mcpServers ?? {});
      expect(serverNames).toHaveLength(2);
    });

    it("handles deleting from empty config", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        agent_settings: {
          mcp_config: null,
        },
      } as any);

      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      const { result } = renderHook(() => useDeleteMcpServer(), {
        wrapper: createWrapper(),
      });

      result.current.mutate("sse-0");

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(saveSettingsSpy).toHaveBeenCalled();
    });
  });

  describe("useUpdateMcpServer", () => {
    it("updates the correct server URL", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        agent_settings: {
          mcp_config: {
            mcpServers: {
              myserver: { url: "https://old-url.com", transport: "sse" },
            },
          },
        },
      } as any);

      const saveSettingsSpy = vi
        .spyOn(SettingsService, "saveSettings")
        .mockResolvedValue(true);

      const { result } = renderHook(() => useUpdateMcpServer(), {
        wrapper: createWrapper(),
      });

      // Use hyphen separator as per implementation
      result.current.mutate({
        serverId: "sse-0",
        server: {
          type: "sse",
          url: "https://new-url.com",
        },
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

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
    });
  });

  describe("error handling", () => {
    it("handles getSettings failure", async () => {
      vi.spyOn(SettingsService, "getSettings").mockRejectedValue(
        new Error("Network error"),
      );

      const { result } = renderHook(() => useAddMcpServer(), {
        wrapper: createWrapper(),
      });

      result.current.mutate({ type: "sse", url: "https://server.com" });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toBeDefined();
    });

    it("handles saveSettings failure", async () => {
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        agent_settings: { mcp_config: null },
      } as any);

      vi.spyOn(SettingsService, "saveSettings").mockRejectedValue(
        new Error("Save failed"),
      );

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
