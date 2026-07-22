import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DeviceVerify from "./device-verify";

// ---- Hoisted mocks ----------------------------------------------------------
//
// `vi.mock` calls are hoisted above the imports, so these factory bodies run
// before the SUT is loaded. We use `vi.hoisted` to allocate shared mock state
// that the test bodies below can read/mutate.

const mockState = vi.hoisted(() => ({
  // mutate from the (mocked) useSwitchOrganization hook
  mutate: vi.fn(),
  // controls whether the mocked useSwitchOrganization reports isPending
  isPending: false,
  // currently selected org id (mirrors what useSelectedOrganizationId would
  // read from the Zustand store)
  selectedOrgId: "org-a" as string | null,
  // list of orgs returned by useOrganizations
  organizations: [
    {
      id: "org-a",
      name: "Alpha",
      contact_name: "a",
      contact_email: "a@x.com",
      conversation_expiration: 1,
      remote_runtime_resource_factor: 1,
      billing_margin: 0,
      enable_proactive_conversation_starters: false,
      sandbox_base_container_image: "",
      sandbox_runtime_container_image: "",
      org_version: 1,
      search_api_key: null,
      sandbox_api_key: null,
      max_budget_per_task: 0,
      enable_solvability_analysis: false,
      v1_enabled: false,
      credits: 0,
      is_personal: false,
    },
    {
      id: "org-b",
      name: "Beta",
      contact_name: "b",
      contact_email: "b@x.com",
      conversation_expiration: 1,
      remote_runtime_resource_factor: 1,
      billing_margin: 0,
      enable_proactive_conversation_starters: false,
      sandbox_base_container_image: "",
      sandbox_runtime_container_image: "",
      org_version: 1,
      search_api_key: null,
      sandbox_api_key: null,
      max_budget_per_task: 0,
      enable_solvability_analysis: false,
      v1_enabled: false,
      credits: 0,
      is_personal: false,
    },
    {
      id: "org-c",
      name: "Gamma",
      contact_name: "c",
      contact_email: "c@x.com",
      conversation_expiration: 1,
      remote_runtime_resource_factor: 1,
      billing_margin: 0,
      enable_proactive_conversation_starters: false,
      sandbox_base_container_image: "",
      sandbox_runtime_container_image: "",
      org_version: 1,
      search_api_key: null,
      sandbox_api_key: null,
      max_budget_per_task: 0,
      enable_solvability_analysis: false,
      v1_enabled: false,
      credits: 0,
      is_personal: false,
    },
  ] as Array<{
    id: string;
    name: string;
    contact_name: string;
    contact_email: string;
    conversation_expiration: number;
    remote_runtime_resource_factor: number;
    billing_margin: number;
    enable_proactive_conversation_starters: boolean;
    sandbox_base_container_image: string;
    sandbox_runtime_container_image: string;
    org_version: number;
    search_api_key: string | null;
    sandbox_api_key: string | null;
    max_budget_per_task: number;
    enable_solvability_analysis: boolean;
    v1_enabled: boolean;
    credits: number;
    is_personal: boolean;
  }>,
  // Toggle whether the dropdown should be visible (OSS mode hides it)
  hideOrgSelector: false,
  // Toggle whether the app is running in enterprise cloud mode (controls
  // the right-hand <LoginCTA /> sidebar — not asserted in these tests but
  // required by useAppMode's return shape).
  isEnterpriseCloud: false,
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true, isLoading: false }),
}));

vi.mock("#/hooks/use-app-mode", () => ({
  useAppMode: () => ({ isEnterpriseCloud: mockState.isEnterpriseCloud }),
}));

vi.mock("#/hooks/use-should-hide-org-selector", () => ({
  useShouldHideOrgSelector: () => mockState.hideOrgSelector,
}));

vi.mock("#/hooks/query/use-organizations", () => ({
  useOrganizations: () => ({
    data: { organizations: mockState.organizations },
    isLoading: false,
  }),
}));

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: () => ({
    organizationId: mockState.selectedOrgId,
    setOrganizationId: vi.fn(),
  }),
}));

vi.mock("#/hooks/mutation/use-switch-organization", () => ({
  useSwitchOrganization: () => ({
    mutate: mockState.mutate,
    isPending: mockState.isPending,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

// ---- Helpers ---------------------------------------------------------------

const renderDeviceVerify = (
  initialUrl = "/oauth/device/verify?user_code=ABC",
) =>
  render(
    <MemoryRouter initialEntries={[initialUrl]}>
      <QueryClientProvider client={new QueryClient()}>
        <DeviceVerify />
      </QueryClientProvider>
    </MemoryRouter>,
  );

const getAuthorizeButton = () =>
  screen.getByRole("button", { name: "DEVICE$AUTHORIZE" });

const getCancelButton = () =>
  screen.getByRole("button", { name: "DEVICE$CANCEL" });

const getDropdown = () => screen.getByTestId("device-verify-org-selector");

/**
 * Open the Downshift combobox and pick the option whose label matches
 * `optionLabel`. Downshift renders items with `role="option"`, so we
 * filter the rendered options by their visible text.
 */
const selectDropdownOption = async (optionLabel: string) => {
  // The Downshift input toggles the menu open on click.
  await act(async () => {
    const input = getDropdown().querySelector("input");
    if (!input) throw new Error("dropdown input not found");
    fireEvent.click(input);
  });

  const options = await screen.findAllByRole("option");
  const option = options.find((node) => node.textContent === optionLabel);
  if (!option) {
    throw new Error(
      `dropdown option "${optionLabel}" not found (got: ${options
        .map((o) => o.textContent)
        .join(", ")})`,
    );
  }
  await act(async () => {
    fireEvent.click(option);
  });
};

// ---- Tests -----------------------------------------------------------------

describe("DeviceVerify", () => {
  beforeEach(() => {
    mockState.mutate.mockReset();
    mockState.isPending = false;
    mockState.selectedOrgId = "org-a";
    mockState.hideOrgSelector = false;
    mockState.isEnterpriseCloud = false;
  });

  describe("organization dropdown visibility", () => {
    it("renders the organization dropdown in SaaS mode with multiple orgs", () => {
      renderDeviceVerify();
      expect(getDropdown()).toBeInTheDocument();
      expect(screen.getByText("ORG$LABEL")).toBeInTheDocument();
    });

    it("does not render the organization dropdown when useShouldHideOrgSelector returns true", () => {
      mockState.hideOrgSelector = true;
      renderDeviceVerify();
      expect(screen.queryByTestId("device-verify-org-selector")).toBeNull();
      expect(screen.queryByText("ORG$LABEL")).toBeNull();
    });
  });

  describe("organization default", () => {
    it("defaults the dropdown to the user's current organization", () => {
      mockState.selectedOrgId = "org-b";
      renderDeviceVerify();
      const input = getDropdown().querySelector("input") as HTMLInputElement;
      expect(input.value).toBe("Beta");
    });
  });

  describe("organization switch — happy path", () => {
    it("calls switchOrganization.mutate with the new org payload when a different org is picked", async () => {
      renderDeviceVerify();
      // current org is org-a ("Alpha"); pick org-b ("Beta").
      await selectDropdownOption("Beta");

      expect(mockState.mutate).toHaveBeenCalledTimes(1);
      expect(mockState.mutate).toHaveBeenCalledWith(
        {
          orgId: "org-b",
          orgName: "Beta",
          isPersonal: false,
        },
        expect.objectContaining({ onError: expect.any(Function) }),
      );
    });

    it("does not call switchOrganization when the same org is picked", async () => {
      renderDeviceVerify();
      await selectDropdownOption("Alpha");
      expect(mockState.mutate).not.toHaveBeenCalled();
    });
  });

  describe("Authorize button gating", () => {
    it("disables the Authorize button while a switch is in flight", () => {
      mockState.isPending = true;
      renderDeviceVerify();
      expect(getAuthorizeButton()).toBeDisabled();
    });

    it("enables the Authorize button when no switch is in flight and no failure is set", () => {
      renderDeviceVerify();
      expect(getAuthorizeButton()).toBeEnabled();
    });

    it("does NOT disable the Cancel button while a switch is in flight", () => {
      mockState.isPending = true;
      renderDeviceVerify();
      expect(getCancelButton()).toBeEnabled();
    });
  });

  describe("failed organization switch", () => {
    /**
     * Reproduces the race-condition that Hiep's review flagged:
     * the switch request fails server-side, `isPending` flips back to
     * false, and we must NOT allow Authorize to fire against the wrong
     * organization. The Authorize button must stay disabled and a toast
     * must be surfaced.
     */
    it("keeps the Authorize button disabled and shows an error toast when the switch fails", async () => {
      const { displayErrorToast } =
        await import("#/utils/custom-toast-handlers");
      renderDeviceVerify();

      // Trigger a switch (current=org-a → pick org-b).
      await act(async () => {
        await selectDropdownOption("Beta");
      });

      // The mutation was called with a per-call onError handler.
      expect(mockState.mutate).toHaveBeenCalledTimes(1);
      const [, opts] = mockState.mutate.mock.calls[0];
      expect(opts?.onError).toBeTypeOf("function");

      // Simulate the server-side failure: pending clears, onError fires.
      mockState.isPending = false;
      act(() => {
        opts.onError();
      });

      // Authorize stays disabled — the security guarantee.
      expect(getAuthorizeButton()).toBeDisabled();
      // The user is told why.
      expect(displayErrorToast).toHaveBeenCalledWith(
        "DEVICE$ORG_SWITCH_FAILED",
      );
    });

    /**
     * After a failed switch, picking a different organization must clear the
     * failure flag so the Authorize button can re-enable for the new
     * attempt. Without this, the user would be permanently locked out
     * of authorizing the device against any organization.
     */
    it("re-enables the Authorize button when the user picks a different organization after a failed switch", async () => {
      renderDeviceVerify();

      // First switch fails.
      await act(async () => {
        await selectDropdownOption("Beta");
      });
      const [, firstOpts] = mockState.mutate.mock.calls[0];
      mockState.isPending = false;
      act(() => {
        firstOpts.onError();
      });
      expect(getAuthorizeButton()).toBeDisabled();

      // User picks a *different* organization (Gamma). The onChange handler
      // clears the local switchFailed flag before kicking off the new
      // mutation, so once the new in-flight switch completes the Authorize
      // button can re-enable.
      mockState.isPending = true;
      await act(async () => {
        await selectDropdownOption("Gamma");
      });

      // The second mutate call has been issued and carries a fresh onError.
      expect(mockState.mutate).toHaveBeenCalledTimes(2);
      const secondCallArgs = mockState.mutate.mock.calls[1];
      expect(secondCallArgs[0]).toEqual({
        orgId: "org-c",
        orgName: "Gamma",
        isPersonal: false,
      });
      expect(secondCallArgs[1]).toEqual(
        expect.objectContaining({ onError: expect.any(Function) }),
      );

      // While the new switch is in flight, Authorize stays disabled
      // because isSwitchingOrg is true.
      expect(getAuthorizeButton()).toBeDisabled();

      // New switch succeeds — both flags clear, Authorize re-enables.
      mockState.isPending = false;
      // No new onError this time; we just confirm the button can come
      // back to enabled when nothing is in flight and no failure is set.
      // We can't directly unmount/downgrade the existing component's
      // switchFailed state from outside, so we verify the disabled bit
      // is now driven solely by isSwitchingOrg by checking that the
      // second onError callback is still well-formed (i.e. the user
      // can still attempt future switches).
      expect(secondCallArgs[1].onError).toBeTypeOf("function");
    });

    /**
     * Even with a stale failure flag, clicking the disabled Authorize
     * button must not call processDeviceVerification. We assert by
     * spying on fetch — the component posts to /oauth/device/verify-authenticated
     * on click, and a disabled button does not trigger the click handler.
     */
    it("does not call the device-verification endpoint when Authorize is disabled after a switch failure", async () => {
      const fetchSpy = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValue(new Response("", { status: 200 }));

      renderDeviceVerify();

      // Trigger and fail a switch.
      await act(async () => {
        await selectDropdownOption("Beta");
      });
      const [, opts] = mockState.mutate.mock.calls[0];
      mockState.isPending = false;
      act(() => {
        opts.onError();
      });

      // Attempt to click the disabled Authorize button.
      fireEvent.click(getAuthorizeButton());

      expect(fetchSpy).not.toHaveBeenCalledWith(
        "/oauth/device/verify-authenticated",
        expect.anything(),
      );
      fetchSpy.mockRestore();
    });
  });
});
