import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { GitOrgRow } from "#/components/features/org/git-org-row";
import type { GitOrg } from "#/hooks/organizations/use-git-conversation-routing";

const createOrg = (overrides: Partial<GitOrg> = {}): GitOrg => ({
  id: "1",
  provider: "GitHub",
  name: "TestOrg",
  status: "unclaimed",
  ...overrides,
});

describe("GitOrgRow", () => {
  it("renders the provider and organization name", () => {
    // Arrange & Act
    renderWithProviders(
      <GitOrgRow
        org={createOrg({ provider: "GitLab", name: "MyOrg" })}
        isLast={false}
        onClaim={vi.fn()}
        onDisconnect={vi.fn()}
      />,
    );

    // Assert
    expect(screen.getByTestId("org-row-1")).toHaveTextContent("GitLab/MyOrg");
  });

  it("renders a claim button for the organization", () => {
    // Arrange & Act
    renderWithProviders(
      <GitOrgRow
        org={createOrg()}
        isLast={false}
        onClaim={vi.fn()}
        onDisconnect={vi.fn()}
      />,
    );

    // Assert
    expect(screen.getByTestId("claim-button-1")).toBeInTheDocument();
  });
});
