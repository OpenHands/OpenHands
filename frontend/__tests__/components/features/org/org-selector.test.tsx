import { screen, render, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { OrgSelector } from "#/components/features/org/org-selector";
import { organizationService } from "#/api/organization-service/organization-service.api";
import * as ToastHandlers from "#/utils/custom-toast-handlers";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import {
  MOCK_PERSONAL_ORG,
  MOCK_TEAM_ORG_ACME,
  createMockOrganization,
} from "#/mocks/org-handlers";

vi.mock("react-router", () => ({
  useRevalidator: () => ({ revalidate: vi.fn() }),
  useNavigate: () => vi.fn(),
  useLocation: () => ({ pathname: "/" }),
  useMatch: () => null,
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

// Mock useConfig to return SaaS mode (organizations are a SaaS-only feature)
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: { app_mode: "saas" } }),
}));

vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string, params?: Record<string, string>) => {
        const translations: Record<string, string> = {
          "ORG$SELECT_ORGANIZATION_PLACEHOLDER": "Please select an organization",
          "ORG$PERSONAL_WORKSPACE": "Personal Workspace",
          "ORG$SWITCHED_TO_ORGANIZATION": `You have switched to organization: ${params?.name ?? ""}`,
          "ORG$SWITCHED_TO_PERSONAL_WORKSPACE":
            "You have switched to your personal workspace.",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: vi.fn(),
      },
    }),
  };
});

const renderOrgSelector = () =>
  render(<OrgSelector />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });

describe("OrgSelector", () => {
  beforeEach(() => {
    useSelectedOrganizationStore.setState({ organizationId: null });
  });
  it("should not render when user only has a personal workspace", async () => {
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_PERSONAL_ORG],
      currentOrgId: MOCK_PERSONAL_ORG.id,
    });

    const { container } = renderOrgSelector();

    await waitFor(() => {
      expect(container).toBeEmptyDOMElement();
    });
  });

  it("should render when user only has a team organization", async () => {
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });

    const { container } = renderOrgSelector();

    await waitFor(() => {
      expect(container).not.toBeEmptyDOMElement();
    });
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should show a loading indicator when fetching organizations", () => {
    vi.spyOn(organizationService, "getOrganizations").mockImplementation(
      () => new Promise(() => {}), // never resolves
    );

    renderOrgSelector();

    // The dropdown trigger should be disabled while loading
    const trigger = screen.getByTestId("dropdown-trigger");
    expect(trigger).toBeDisabled();
  });

  it("should select the first organization after orgs are loaded", async () => {
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_PERSONAL_ORG, MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_PERSONAL_ORG.id,
    });

    renderOrgSelector();

    // The combobox input should show the first org name
    await waitFor(() => {
      const input = screen.getByRole("combobox");
      expect(input).toHaveValue("Personal Workspace");
    });
  });

  it("should show all options when dropdown is opened", async () => {
    const user = userEvent.setup();
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [
        MOCK_PERSONAL_ORG,
        MOCK_TEAM_ORG_ACME,
        createMockOrganization("3", "Test Organization", 500),
      ],
      currentOrgId: MOCK_PERSONAL_ORG.id,
    });

    renderOrgSelector();

    // Wait for the selector to be populated with the first organization
    await waitFor(() => {
      const input = screen.getByRole("combobox");
      expect(input).toHaveValue("Personal Workspace");
    });

    // Click the trigger to open dropdown
    const trigger = screen.getByTestId("dropdown-trigger");
    await user.click(trigger);

    // Verify all 3 options are visible
    const listbox = await screen.findByRole("listbox");
    const options = within(listbox).getAllByRole("option");

    expect(options).toHaveLength(3);
    expect(options[0]).toHaveTextContent("Personal Workspace");
    expect(options[1]).toHaveTextContent("Acme Corp");
    expect(options[2]).toHaveTextContent("Test Organization");
  });

  it("should call switchOrganization API when selecting a different organization", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_PERSONAL_ORG, MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_PERSONAL_ORG.id,
    });
    const switchOrgSpy = vi
      .spyOn(organizationService, "switchOrganization")
      .mockResolvedValue(MOCK_TEAM_ORG_ACME);

    renderOrgSelector();

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toHaveValue("Personal Workspace");
    });

    // Act
    const trigger = screen.getByTestId("dropdown-trigger");
    await user.click(trigger);
    const listbox = await screen.findByRole("listbox");
    const acmeOption = within(listbox).getByText("Acme Corp");
    await user.click(acmeOption);

    // Assert
    expect(switchOrgSpy).toHaveBeenCalledWith({ orgId: MOCK_TEAM_ORG_ACME.id });
  });

  it("should show loading state while switching organizations", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_PERSONAL_ORG, MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_PERSONAL_ORG.id,
    });
    vi.spyOn(organizationService, "switchOrganization").mockImplementation(
      () => new Promise(() => {}), // never resolves to keep loading state
    );

    renderOrgSelector();

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toHaveValue("Personal Workspace");
    });

    // Act
    const trigger = screen.getByTestId("dropdown-trigger");
    await user.click(trigger);
    const listbox = await screen.findByRole("listbox");
    const acmeOption = within(listbox).getByText("Acme Corp");
    await user.click(acmeOption);

    // Assert
    await waitFor(() => {
      expect(screen.getByTestId("dropdown-trigger")).toBeDisabled();
    });
  });

  it("should display toast with organization name when switching to a team organization", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_PERSONAL_ORG, MOCK_TEAM_ORG_ACME],
      currentOrgId: MOCK_PERSONAL_ORG.id,
    });
    vi.spyOn(organizationService, "switchOrganization").mockResolvedValue(
      MOCK_TEAM_ORG_ACME,
    );
    const displaySuccessToastSpy = vi.spyOn(
      ToastHandlers,
      "displaySuccessToast",
    );

    renderOrgSelector();

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toHaveValue("Personal Workspace");
    });

    // Act
    const trigger = screen.getByTestId("dropdown-trigger");
    await user.click(trigger);
    const listbox = await screen.findByRole("listbox");
    const acmeOption = within(listbox).getByText("Acme Corp");
    await user.click(acmeOption);

    // Assert
    await waitFor(() => {
      expect(displaySuccessToastSpy).toHaveBeenCalledWith(
        "You have switched to organization: Acme Corp",
      );
    });
  });

  it("should display toast for personal workspace when switching to personal workspace", async () => {
    // Arrange
    const user = userEvent.setup();
    // Pre-set the store to have team org selected
    useSelectedOrganizationStore.setState({
      organizationId: MOCK_TEAM_ORG_ACME.id,
    });
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue({
      items: [MOCK_TEAM_ORG_ACME, MOCK_PERSONAL_ORG],
      currentOrgId: MOCK_TEAM_ORG_ACME.id,
    });
    vi.spyOn(organizationService, "switchOrganization").mockResolvedValue(
      MOCK_PERSONAL_ORG,
    );
    const displaySuccessToastSpy = vi.spyOn(
      ToastHandlers,
      "displaySuccessToast",
    );

    renderOrgSelector();

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toHaveValue("Acme Corp");
    });

    // Act
    const trigger = screen.getByTestId("dropdown-trigger");
    await user.click(trigger);
    const listbox = await screen.findByRole("listbox");
    const personalOption = within(listbox).getByText("Personal Workspace");
    await user.click(personalOption);

    // Assert
    await waitFor(() => {
      expect(displaySuccessToastSpy).toHaveBeenCalledWith(
        "You have switched to your personal workspace.",
      );
    });
  });
});
