import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useCloudCurrentUserId } from "#/hooks/query/use-cloud-current-user-id";

const getCloudOrganizationMeMock = vi.fn();
const useAllCloudOrganizationsMock = vi.fn();

vi.mock("#/api/cloud/organization-service.api", () => ({
  getCloudOrganizationMe: (...args: unknown[]) =>
    getCloudOrganizationMeMock(...args),
}));
vi.mock("#/hooks/query/use-cloud-organizations", () => ({
  useAllCloudOrganizations: () => useAllCloudOrganizationsMock(),
}));

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  }
  return Wrapper;
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  getCloudOrganizationMeMock.mockReset();
  useAllCloudOrganizationsMock.mockReset();
  useAllCloudOrganizationsMock.mockReturnValue({
    [cloudBackend.id]: {
      backend: cloudBackend,
      isLoading: false,
      orgs: [
        { id: "org-personal", name: "Personal" },
        { id: "org-2", name: "Acme Inc" },
      ],
    },
  });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useCloudCurrentUserId", () => {
  it("uses active.orgId for /me when the active backend is this cloud backend", async () => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });
    getCloudOrganizationMeMock.mockResolvedValue({
      orgId: "org-2",
      userId: "user-X",
    });

    const { result } = renderHook(() => useCloudCurrentUserId(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current[cloudBackend.id]?.userId).toBe("user-X");
    });

    // /me must be invoked with the ACTIVE org (org-2), not the first org.
    expect(getCloudOrganizationMeMock).toHaveBeenCalledOnce();
    expect(getCloudOrganizationMeMock).toHaveBeenCalledWith(
      "org-2",
      cloudBackend,
    );
  });

  it("falls back to the first org for /me when no org is selected yet", async () => {
    // Active is bundled local — `prod` is registered but not active.
    getCloudOrganizationMeMock.mockResolvedValue({
      orgId: "org-personal",
      userId: "user-X",
    });

    const { result } = renderHook(() => useCloudCurrentUserId(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current[cloudBackend.id]?.userId).toBe("user-X");
    });

    // Sentinel = first org.
    expect(getCloudOrganizationMeMock).toHaveBeenCalledWith(
      "org-personal",
      cloudBackend,
    );
  });

  it("issues exactly one /me call per cloud backend (not one per org)", async () => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });
    getCloudOrganizationMeMock.mockResolvedValue({
      orgId: "org-2",
      userId: "user-X",
    });

    renderHook(() => useCloudCurrentUserId(), { wrapper: makeWrapper() });

    await waitFor(() => {
      expect(getCloudOrganizationMeMock).toHaveBeenCalledTimes(1);
    });
  });
});
