import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ReactNode } from "react";
import { useSaveOrgAppSettings } from "#/hooks/mutation/use-save-org-app-settings";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { ORGANIZATION_SETTINGS_KEY } from "#/hooks/query/query-keys";

vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    saveOrganizationAppSettings: vi.fn(),
  },
}));

vi.mock("#/hooks/query/query-keys", () => ({
  ORGANIZATION_SETTINGS_KEY: ["organization-settings"],
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const mockResponse = {
  enable_proactive_conversation_starters: true,
  max_budget_per_task: 100,
  registered_marketplaces: [
    {
      name: "test-marketplace",
      source: "github:owner/repo",
      auto_load: true,
    },
  ],
  updated_at: "2024-01-01T00:00:00Z",
};

describe("useSaveOrgAppSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("accepts { orgId, settings } params", async () => {
    vi.mocked(organizationService.saveOrganizationAppSettings).mockResolvedValue(
      mockResponse,
    );

    const { result } = renderHook(() => useSaveOrgAppSettings(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      orgId: "org-123",
      settings: {
        registered_marketplaces: [
          { name: "test", source: "github:owner/repo" },
        ],
      },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(
      organizationService.saveOrganizationAppSettings,
    ).toHaveBeenCalledWith({
      orgId: "org-123",
      settings: {
        registered_marketplaces: [{ name: "test", source: "github:owner/repo" }],
      },
    });
  });

  it("sends correct orgId to service", async () => {
    vi.mocked(organizationService.saveOrganizationAppSettings).mockResolvedValue(
      mockResponse,
    );

    const { result } = renderHook(() => useSaveOrgAppSettings(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      orgId: "org-456",
      settings: {
        enable_proactive_conversation_starters: false,
      },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(
      organizationService.saveOrganizationAppSettings,
    ).toHaveBeenCalledWith({
      orgId: "org-456",
      settings: {
        enable_proactive_conversation_starters: false,
      },
    });
  });

  it("handles save error", async () => {
    const error = new Error("Failed to save");
    vi.mocked(organizationService.saveOrganizationAppSettings).mockRejectedValue(
      error,
    );

    const { result } = renderHook(() => useSaveOrgAppSettings(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      orgId: "org-123",
      settings: {},
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toBe(error);
  });

  it("uses mutateAsync for promise-based save", async () => {
    vi.mocked(organizationService.saveOrganizationAppSettings).mockResolvedValue(
      mockResponse,
    );

    const { result } = renderHook(() => useSaveOrgAppSettings(), {
      wrapper: createWrapper(),
    });

    const promise = result.current.mutateAsync({
      orgId: "org-123",
      settings: {
        max_budget_per_task: 50,
      },
    });

    await expect(promise).resolves.toEqual(mockResponse);

    expect(
      organizationService.saveOrganizationAppSettings,
    ).toHaveBeenCalledWith({
      orgId: "org-123",
      settings: {
        max_budget_per_task: 50,
      },
    });
  });

});
