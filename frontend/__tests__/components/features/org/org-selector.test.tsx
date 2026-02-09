import { screen, render, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { OrgSelector } from "#/components/features/org/org-selector";
import { organizationService } from "#/api/organization-service/organization-service.api";
import {
  MOCK_PERSONAL_ORG,
  MOCK_TEAM_ORG_ACME,
  createMockOrganization,
} from "#/mocks/org-handlers";

vi.mock("react-router", () => ({
  useRevalidator: () => ({ revalidate: vi.fn() }),
}));

vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          "ORG$SELECT_ORGANIZATION_PLACEHOLDER": "Please select an organization",
          "ORG$PERSONAL_WORKSPACE": "Personal Workspace",
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
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue([
      MOCK_PERSONAL_ORG,
      MOCK_TEAM_ORG_ACME,
    ]);

    renderOrgSelector();

    // The combobox input should show the first org name
    await waitFor(() => {
      const input = screen.getByRole("combobox");
      expect(input).toHaveValue("Personal Workspace");
    });
  });

  it("should show all options when dropdown is opened", async () => {
    const user = userEvent.setup();
    vi.spyOn(organizationService, "getOrganizations").mockResolvedValue([
      MOCK_PERSONAL_ORG,
      MOCK_TEAM_ORG_ACME,
      createMockOrganization("3", "Test Organization", 500),
    ]);

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
});
