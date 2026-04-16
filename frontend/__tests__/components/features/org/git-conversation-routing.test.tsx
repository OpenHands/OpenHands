import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { GitConversationRouting } from "#/components/features/org/git-conversation-routing";

const mockMutate = vi.fn();
const mockDisconnectMutate = vi.fn();

vi.mock("#/hooks/query/use-git-organizations", () => ({
  useUserGitOrganizations: () => ({
    data: {
      provider: "github",
      organizations: ["OpenHands", "AcmeCo"],
    },
    isLoading: false,
  }),
  useGitClaims: () => ({
    data: [
      {
        id: "claim-1",
        org_id: "org-1",
        provider: "github",
        git_organization: "OpenHands",
        claimed_by: "user-1",
        claimed_at: "2026-01-01T00:00:00",
      },
    ],
    isLoading: false,
  }),
}));

vi.mock("#/hooks/mutation/use-claim-git-org", () => ({
  useClaimGitOrg: () => ({
    mutate: mockMutate,
  }),
}));

vi.mock("#/hooks/mutation/use-disconnect-git-org", () => ({
  useDisconnectGitOrg: () => ({
    mutate: mockDisconnectMutate,
  }),
}));

describe("GitConversationRouting", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should render organizations from API data", () => {
    renderWithProviders(<GitConversationRouting />);

    expect(
      screen.getByTestId("org-row-github:openhands"),
    ).toHaveTextContent("github/OpenHands");
    expect(
      screen.getByTestId("org-row-github:acmeco"),
    ).toHaveTextContent("github/AcmeCo");
  });

  it("should show claimed org with 'Claimed' label", () => {
    renderWithProviders(<GitConversationRouting />);

    const claimedButton = screen.getByTestId("claim-button-github:openhands");
    expect(claimedButton).toHaveTextContent("ORG$CLAIMED");
  });

  it("should show unclaimed orgs with 'Claim' label", () => {
    renderWithProviders(<GitConversationRouting />);

    expect(
      screen.getByTestId("claim-button-github:acmeco"),
    ).toHaveTextContent("ORG$CLAIM");
  });

  it("should call claim mutation when clicking claim on unclaimed org", async () => {
    renderWithProviders(<GitConversationRouting />);
    const user = userEvent.setup();

    await user.click(screen.getByTestId("claim-button-github:acmeco"));

    expect(mockMutate).toHaveBeenCalledWith(
      { provider: "github", gitOrganization: "AcmeCo" },
      expect.objectContaining({ onSettled: expect.any(Function) }),
    );
  });

  it("should call disconnect mutation when clicking disconnect on claimed org", async () => {
    renderWithProviders(<GitConversationRouting />);
    const user = userEvent.setup();

    await user.click(screen.getByTestId("claim-button-github:openhands"));

    expect(mockDisconnectMutate).toHaveBeenCalledWith(
      { claimId: "claim-1" },
      expect.objectContaining({ onSettled: expect.any(Function) }),
    );
  });

});
