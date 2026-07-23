import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactNode } from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useOrgUsageStats } from "#/hooks/query/use-org-usage-stats";
import { organizationService } from "#/api/organization-service/organization-service.api";

vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    getUsageStats: vi.fn(),
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
    },
  });

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const mockUsageStats = {
  active_users: 0,
  agent_runs: 0,
  usage_conversation_count: 0,
  total_tokens: 0,
  estimated_spend: 0,
  daily_usage: [],
  team_usage: [],
  model_usage: [],
  agent_usage: [],
};

describe("useOrgUsageStats", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("passes timeWindow through without setting days", async () => {
    vi.mocked(useSelectedOrganizationId).mockReturnValue({
      organizationId: "org-123",
      setOrganizationId: vi.fn(),
    });
    vi.mocked(organizationService.getUsageStats).mockResolvedValue(
      mockUsageStats,
    );

    const { result } = renderHook(
      () => useOrgUsageStats({ timeWindow: "ytd" }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(organizationService.getUsageStats).toHaveBeenCalledWith({
      orgId: "org-123",
      days: undefined,
      timeWindow: "ytd",
    });
  });
});
