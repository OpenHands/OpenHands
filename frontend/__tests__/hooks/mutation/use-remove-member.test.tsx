import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useRemoveMember } from "#/hooks/mutation/use-remove-member";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

// Mock the useRevalidator hook from react-router
vi.mock("react-router", () => ({
  useRevalidator: () => ({
    revalidate: vi.fn(),
  }),
}));

describe("useRemoveMember", () => {
  beforeEach(() => {
    useSelectedOrganizationStore.setState({ organizationId: null });
  });

  it("should throw an error when organizationId is null", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        mutations: {
          retry: false,
        },
      },
    });

    const { result } = renderHook(() => useRemoveMember(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      ),
    });

    // Attempt to mutate without organizationId
    result.current.mutate({ userId: "user-123" });

    // Should fail with an error about missing organizationId
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe("Organization ID is required");
  });
});
