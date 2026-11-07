import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useAllCloudOrganizations } from "#/hooks/query/use-cloud-organizations";

const getCloudOrganizationsMock = vi.fn();
const getCurrentCloudApiKeyMock = vi.fn();

vi.mock("#/api/cloud/organization-service.api", () => ({
  getCloudOrganizations: (...args: unknown[]) =>
    getCloudOrganizationsMock(...args),
  getCurrentCloudApiKey: (...args: unknown[]) =>
    getCurrentCloudApiKeyMock(...args),
}));

function cloudBackend(overrides: Partial<Backend> = {}): Backend {
  return {
    id: "prod",
    name: "Production",
    host: "https://app.all-hands.dev",
    apiKey: "bearer-token",
    kind: "cloud",
    ...overrides,
  };
}

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
  getCloudOrganizationsMock.mockReset();
  getCurrentCloudApiKeyMock.mockReset();
  // Legacy/unbound key by default: the unfiltered path the visibility tests
  // care about. Key-scoping has its own coverage in the selector tests.
  getCurrentCloudApiKeyMock.mockResolvedValue({
    orgId: null,
    isLegacyKey: true,
  });
  setRegisteredBackends([cloudBackend()]);
});

describe("useAllCloudOrganizations visibility filtering", () => {
  it("hides orgs the cloud marks is_visible=false", async () => {
    getCloudOrganizationsMock.mockResolvedValue({
      items: [
        { id: "org-team", name: "Acme Inc", is_visible: true },
        { id: "org-personal", name: "Personal", is_visible: false },
      ],
      currentOrgId: "org-team",
    });

    const { result } = renderHook(() => useAllCloudOrganizations(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.prod?.orgs).toHaveLength(1);
    });
    expect(result.current.prod?.orgs.map((o) => o.id)).toEqual(["org-team"]);
  });

  it("keeps orgs when the backend predates is_visible (field absent)", async () => {
    // Backwards compatibility: an older app-server never sends the field,
    // and those orgs must not disappear from the selector.
    getCloudOrganizationsMock.mockResolvedValue({
      items: [
        { id: "org-1", name: "Acme Inc" },
        { id: "org-2", name: "Personal", is_personal: true },
      ],
      currentOrgId: "org-1",
    });

    const { result } = renderHook(() => useAllCloudOrganizations(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.prod?.orgs).toHaveLength(2);
    });
    expect(result.current.prod?.orgs.map((o) => o.id)).toEqual([
      "org-1",
      "org-2",
    ]);
  });

  it("treats an explicit is_visible=true as visible", async () => {
    getCloudOrganizationsMock.mockResolvedValue({
      items: [{ id: "org-1", name: "Acme Inc", is_visible: true }],
      currentOrgId: "org-1",
    });

    const { result } = renderHook(() => useAllCloudOrganizations(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.prod?.orgs).toHaveLength(1);
    });
  });

  it("hides invisible orgs on the cookie-auth path too", async () => {
    setRegisteredBackends([cloudBackend({ authMode: "cookie" })]);
    getCloudOrganizationsMock.mockResolvedValue({
      items: [
        { id: "org-team", name: "Acme Inc" },
        { id: "org-personal", name: "Personal", is_visible: false },
      ],
      currentOrgId: "org-team",
    });

    const { result } = renderHook(() => useAllCloudOrganizations(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.prod?.orgs).toHaveLength(1);
    });
    expect(result.current.prod?.orgs.map((o) => o.id)).toEqual(["org-team"]);
    // The cookie path short-circuits before the key lookup.
    expect(getCurrentCloudApiKeyMock).not.toHaveBeenCalled();
  });

  it("applies visibility before key scoping by org id", async () => {
    // An invisible org must stay hidden even when it is the org the API key
    // is bound to — the key scoping is an intersection, never a re-reveal.
    getCurrentCloudApiKeyMock.mockResolvedValue({
      orgId: "org-personal",
      isLegacyKey: false,
    });
    getCloudOrganizationsMock.mockResolvedValue({
      items: [
        { id: "org-team", name: "Acme Inc" },
        { id: "org-personal", name: "Personal", is_visible: false },
      ],
      currentOrgId: "org-personal",
    });

    const { result } = renderHook(() => useAllCloudOrganizations(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.prod?.isLoading).toBe(false);
    });
    expect(result.current.prod?.orgs).toEqual([]);
  });
});
