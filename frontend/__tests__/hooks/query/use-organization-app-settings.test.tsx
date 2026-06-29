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

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: vi.fn(),
}));

import { useSelectedOrganizationId } from "#/context/use-selected-organization";

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
      scope: "org" as const,
    },
  ],
  updated_at: "2024-01-01T00:00:00Z",
};

describe("useOrganizationAppSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not fetch when orgId is null", async () => {
    vi.mocked(useSelectedOrganizationId).mockReturnValue({ organizationId: null, setOrganizationId: vi.fn() });

    const { result } = renderHook(
      () => useOrganizationAppSettings(),
      { wrapper: createWrapper() },
    );

    expect(organizationService.getOrganizationAppSettings).not.toHaveBeenCalled();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
  });

  it("does not fetch when orgId is undefined", async () => {
    vi.mocked(useSelectedOrganizationId).mockReturnValue({ organizationId: null, setOrganizationId: vi.fn() });

    const { result } = renderHook(
      () => useOrganizationAppSettings(),
      { wrapper: createWrapper() },
    );

    expect(organizationService.getOrganizationAppSettings).not.toHaveBeenCalled();
    expect(result.current.isLoading).toBe(false);
  });

  it("fetches org app settings when orgId is provided", async () => {
    vi.mocked(useSelectedOrganizationId).mockReturnValue({ organizationId: "org-123", setOrganizationId: vi.fn() });
    vi.mocked(organizationService.getOrganizationAppSettings).mockResolvedValue(
      mockResponse,
    );

    const { result } = renderHook(
      () => useOrganizationAppSettings(),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(
      organizationService.getOrganizationAppSettings,
    ).toHaveBeenCalled();
    expect(result.current.data).toEqual(mockResponse);
    expect(result.current.data?.registered_marketplaces).toHaveLength(1);
  });

  it("uses different query keys for different orgs", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    vi.mocked(useSelectedOrganizationId).mockReturnValue({ organizationId: "org-123", setOrganizationId: vi.fn() });
    vi.mocked(organizationService.getOrganizationAppSettings).mockResolvedValue({
      ...mockResponse,
      registered_marketplaces: [{ name: "marketplace-123", source: "github:org/123", auto_load: true, scope: "org" as const }],
    });

    const wrapper1 = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { result: result1, rerender } = renderHook(
      () => useOrganizationAppSettings(),
      { wrapper: wrapper1 },
    );

    await waitFor(() => expect(result1.current.isLoading).toBe(false));
    expect(result1.current.data?.registered_marketplaces?.[0]?.name).toBe("marketplace-123");

    // Change org - should create new query with different key
    vi.mocked(useSelectedOrganizationId).mockReturnValue({ organizationId: "org-456", setOrganizationId: vi.fn() });
    vi.mocked(organizationService.getOrganizationAppSettings).mockResolvedValue({
      ...mockResponse,
      registered_marketplaces: [{ name: "marketplace-456", source: "github:org/456", auto_load: true, scope: "org" as const }],
    });
    rerender();

    await waitFor(() => expect(result1.current.isLoading).toBe(false));
    expect(result1.current.data?.registered_marketplaces?.[0]?.name).toBe("marketplace-456");
  });

  it("handles fetch error", async () => {
    vi.mocked(useSelectedOrganizationId).mockReturnValue({ organizationId: "org-123", setOrganizationId: vi.fn() });
    const error = new Error("Failed to fetch");
    vi.mocked(organizationService.getOrganizationAppSettings).mockRejectedValue(
      error,
    );

    const { result } = renderHook(
      () => useOrganizationAppSettings(),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toBe(error);
  });
});
