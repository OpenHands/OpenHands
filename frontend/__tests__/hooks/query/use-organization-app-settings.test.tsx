import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ReactNode } from "react";
import { useOrganizationAppSettings } from "#/hooks/query/use-organization-app-settings";
import { organizationService } from "#/api/organization-service/organization-service.api";

vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    getOrganizationAppSettings: vi.fn(),
  },
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

describe("useOrganizationAppSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not fetch when orgId is null", async () => {
    const { result } = renderHook(
      () => useOrganizationAppSettings(null),
      { wrapper: createWrapper() },
    );

    expect(organizationService.getOrganizationAppSettings).not.toHaveBeenCalled();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
  });

  it("does not fetch when orgId is undefined", async () => {
    const { result } = renderHook(
      () => useOrganizationAppSettings(undefined as unknown as string | null),
      { wrapper: createWrapper() },
    );

    expect(organizationService.getOrganizationAppSettings).not.toHaveBeenCalled();
    expect(result.current.isLoading).toBe(false);
  });

  it("fetches org app settings when orgId is provided", async () => {
    vi.mocked(organizationService.getOrganizationAppSettings).mockResolvedValue(
      mockResponse,
    );

    const { result } = renderHook(
      () => useOrganizationAppSettings("org-123"),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(
      organizationService.getOrganizationAppSettings,
    ).toHaveBeenCalledWith({ orgId: "org-123" });
    expect(result.current.data).toEqual(mockResponse);
  });

  it("fetches with different orgId when orgId changes", async () => {
    vi.mocked(organizationService.getOrganizationAppSettings).mockResolvedValue(
      mockResponse,
    );

    // First render with null
    const { result: result1 } = renderHook(
      () => useOrganizationAppSettings(null),
      { wrapper: createWrapper() },
    );

    expect(result1.current.isLoading).toBeFalsy();
    expect(
      organizationService.getOrganizationAppSettings,
    ).not.toHaveBeenCalled();

    // Second render with orgId
    const { result: result2 } = renderHook(
      () => useOrganizationAppSettings("org-456"),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result2.current.isLoading).toBeFalsy());

    expect(
      organizationService.getOrganizationAppSettings,
    ).toHaveBeenCalledWith({ orgId: "org-456" });
  });

  it("handles fetch error", async () => {
    const error = new Error("Failed to fetch");
    vi.mocked(organizationService.getOrganizationAppSettings).mockRejectedValue(
      error,
    );

    const { result } = renderHook(
      () => useOrganizationAppSettings("org-123"),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toBe(error);
  });
});
