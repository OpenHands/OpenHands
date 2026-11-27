import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";
import * as orgService from "#/api/cloud/organization-service.api";
import * as activeBackendContext from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";

vi.mock("#/api/cloud/organization-service.api");

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function mockBackend(backend: Backend, orgId: string | null) {
  vi.spyOn(activeBackendContext, "useActiveBackend").mockReturnValue({
    backend,
    orgId,
  });
}

describe("useCanManageOrgProfiles", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("is always true on local backends (no /me call)", () => {
    mockBackend(localBackend, null);
    const { result } = renderHook(() => useCanManageOrgProfiles(), { wrapper });
    expect(result.current).toBe(true);
    expect(orgService.getCloudOrganizationMe).not.toHaveBeenCalled();
  });

  it("uses the server permission and ignores the role (permission grants)", async () => {
    mockBackend(cloudBackend, "org-1");
    // Role says member, but the permission set grants edit — permission wins.
    vi.mocked(orgService.getCloudOrganizationMe).mockResolvedValue({
      orgId: "org-1",
      userId: "u",
      role: "member",
      permissions: ["view_org_settings", "edit_org_settings"],
    });
    const { result } = renderHook(() => useCanManageOrgProfiles(), { wrapper });
    await waitFor(() => expect(result.current).toBe(true));
  });

  it("uses the server permission and ignores the role (permission denies)", async () => {
    mockBackend(cloudBackend, "org-1");
    // Role says owner, but the permission set lacks edit — permission wins.
    vi.mocked(orgService.getCloudOrganizationMe).mockResolvedValue({
      orgId: "org-1",
      userId: "u",
      role: "owner",
      permissions: ["view_org_settings"],
    });
    const { result } = renderHook(() => useCanManageOrgProfiles(), { wrapper });
    await waitFor(() =>
      expect(orgService.getCloudOrganizationMe).toHaveBeenCalled(),
    );
    expect(result.current).toBe(false);
  });

  it("falls back to the role when the app-server omits permissions", async () => {
    mockBackend(cloudBackend, "org-1");
    vi.mocked(orgService.getCloudOrganizationMe).mockResolvedValue({
      orgId: "org-1",
      userId: "u",
      role: "admin",
      permissions: null,
    });
    const { result } = renderHook(() => useCanManageOrgProfiles(), { wrapper });
    await waitFor(() => expect(result.current).toBe(true));
  });

  it("falls back to deny for a member when permissions are absent", async () => {
    mockBackend(cloudBackend, "org-1");
    vi.mocked(orgService.getCloudOrganizationMe).mockResolvedValue({
      orgId: "org-1",
      userId: "u",
      role: "member",
      permissions: null,
    });
    const { result } = renderHook(() => useCanManageOrgProfiles(), { wrapper });
    await waitFor(() =>
      expect(orgService.getCloudOrganizationMe).toHaveBeenCalled(),
    );
    expect(result.current).toBe(false);
  });
});
