import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AxiosError, AxiosHeaders } from "axios";

// Import after mocking
import { organizationService } from "#/api/organization-service/organization-service.api";
import * as ToastHandlers from "#/utils/custom-toast-handlers";
import { useAutoAcceptInvitation } from "#/hooks/use-auto-accept-invitation";

const INVITATION_TOKEN_KEY = "openhands_invitation_token";

// Mock react-router (useInvitation reads search params)
vi.mock("react-router", () => ({
  useSearchParams: () => [
    {
      get: () => null,
      has: () => false,
    },
    vi.fn(),
  ],
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

vi.mock("#/hooks/use-is-on-intermediate-page", () => ({
  useIsOnIntermediatePage: () => false,
}));

const mockSwitchOrganization = vi.fn();
vi.mock("#/hooks/mutation/use-switch-organization", () => ({
  useSwitchOrganization: () => ({ mutate: mockSwitchOrganization }),
}));

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>
    {children}
  </QueryClientProvider>
);

const makeAxiosError = (status: number, detail: string) =>
  new AxiosError("Request failed", String(status), undefined, undefined, {
    status,
    statusText: "",
    headers: {},
    config: { headers: new AxiosHeaders() },
    data: { detail },
  });

describe("useAutoAcceptInvitation", () => {
  beforeEach(() => {
    localStorage.setItem(INVITATION_TOKEN_KEY, "inv-test-token");
  });

  afterEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("should accept the stored invitation token and switch to the org", async () => {
    const acceptSpy = vi
      .spyOn(organizationService, "acceptInvitation")
      .mockResolvedValue({
        success: true,
        org_id: "org-1",
        org_name: "Acme",
        role: "member",
      });
    const successToastSpy = vi.spyOn(ToastHandlers, "displaySuccessToast");

    renderHook(() => useAutoAcceptInvitation(), { wrapper });

    await waitFor(() =>
      expect(acceptSpy).toHaveBeenCalledExactlyOnceWith({
        token: "inv-test-token",
      }),
    );
    await waitFor(() => expect(successToastSpy).toHaveBeenCalled());
    expect(mockSwitchOrganization).toHaveBeenCalledWith({
      orgId: "org-1",
      orgName: "Acme",
      isPersonal: false,
    });
    // Token is cleared after the attempt
    await waitFor(() =>
      expect(localStorage.getItem(INVITATION_TOKEN_KEY)).toBeNull(),
    );
  });

  it("should treat already_member as success, not an error", async () => {
    vi.spyOn(organizationService, "acceptInvitation").mockRejectedValue(
      makeAxiosError(409, "already_member"),
    );
    const successToastSpy = vi.spyOn(ToastHandlers, "displaySuccessToast");
    const errorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");

    renderHook(() => useAutoAcceptInvitation(), { wrapper });

    await waitFor(() => expect(successToastSpy).toHaveBeenCalled());
    expect(errorToastSpy).not.toHaveBeenCalled();
    expect(mockSwitchOrganization).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(localStorage.getItem(INVITATION_TOKEN_KEY)).toBeNull(),
    );
  });

  it("should show an error toast for an expired invitation", async () => {
    vi.spyOn(organizationService, "acceptInvitation").mockRejectedValue(
      makeAxiosError(400, "invitation_expired"),
    );
    const errorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");

    renderHook(() => useAutoAcceptInvitation(), { wrapper });

    await waitFor(() => expect(errorToastSpy).toHaveBeenCalled());
    expect(mockSwitchOrganization).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(localStorage.getItem(INVITATION_TOKEN_KEY)).toBeNull(),
    );
  });

  it("should only attempt acceptance once per token", async () => {
    const acceptSpy = vi
      .spyOn(organizationService, "acceptInvitation")
      .mockResolvedValue({
        success: true,
        org_id: "org-1",
        org_name: "Acme",
        role: "member",
      });

    const { rerender } = renderHook(() => useAutoAcceptInvitation(), {
      wrapper,
    });
    rerender();
    rerender();

    await waitFor(() => expect(acceptSpy).toHaveBeenCalledTimes(1));
  });
});
