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

  // SDK 1.31.x dropped the ``mcpServers`` wrapper from ``model_dump``, so the
  // settings endpoint now returns a flat server map. ``parseMcpConfig`` still
  // accepts the legacy wrapped shape, and ``toSdkMcpConfig`` now emits the
  // flat shape. Tests below cover both directions.

  describe("useAddMcpServer", () => {
    it("fetches fresh settings at mutation time", async () => {
      const getSettingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue({
          agent_settings: {
            mcp_config: {
              existing: { url: "https://existing.com", transport: "sse" },
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
          mcp_config: expect.objectContaining({
            existing: expect.objectContaining({ url: "https://existing.com" }),
          }),
        },
      });
      // Should not wrap in the legacy ``mcpServers`` envelope.
      const savedConfig = (
        saveSettingsSpy.mock.calls[0][0] as {
          agent_settings_diff: { mcp_config: Record<string, unknown> };
        }
      ).agent_settings_diff.mcp_config;
      expect(savedConfig).not.toHaveProperty("mcpServers");
    });

    it("also accepts the legacy wrapped shape on read", async () => {
      // Backwards-compat: a server persisted before the SDK 1.31.x bump may
      // still come back as ``{ mcpServers: {...} }`` if the value has not yet
      // been rewritten by the backend. The frontend must still parse it.
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
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

      result.current.mutate({ type: "sse", url: "https://new-server.com" });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(saveSettingsSpy).toHaveBeenCalledWith({
        agent_settings_diff: {
          mcp_config: expect.objectContaining({
            existing: expect.objectContaining({ url: "https://existing.com" }),
          }),
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
            sse: {
              url: "https://first-server.com",
              transport: "sse",
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
            sse: {
              url: "https://server.com",
              transport: "sse",
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
            server1: { url: "https://server1.com", transport: "sse" },
            server2: { url: "https://server2.com", transport: "sse" },
            server3: { url: "https://server3.com", transport: "sse" },
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
          mcp_config: Record<string, unknown> | null;
        };
      };
      const savedConfig = savedPayload.agent_settings_diff.mcp_config;
      const serverNames = Object.keys(savedConfig ?? {});
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
            myserver: { url: "https://old-url.com", transport: "sse" },
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
          mcp_config: Record<string, { url: string }>;
        };
      };
      const savedConfig = savedPayload.agent_settings_diff.mcp_config;
      const serverUrls = Object.values(savedConfig).map((s) => s.url);
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
