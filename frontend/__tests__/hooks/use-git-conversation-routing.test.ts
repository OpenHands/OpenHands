import { renderHook, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

const mockClaimMutate = vi.fn();
const mockDisconnectMutate = vi.fn();

vi.mock("#/hooks/query/use-git-organizations", () => ({
  useUserGitOrganizations: vi.fn(),
  useGitClaims: vi.fn(),
}));

vi.mock("#/hooks/mutation/use-claim-git-org", () => ({
  useClaimGitOrg: () => ({ mutate: mockClaimMutate }),
}));

vi.mock("#/hooks/mutation/use-disconnect-git-org", () => ({
  useDisconnectGitOrg: () => ({ mutate: mockDisconnectMutate }),
}));

import { useGitConversationRouting } from "#/hooks/organizations/use-git-conversation-routing";
import {
  useUserGitOrganizations,
  useGitClaims,
} from "#/hooks/query/use-git-organizations";

const mockUseUserGitOrganizations = vi.mocked(useUserGitOrganizations);
const mockUseGitClaims = vi.mocked(useGitClaims);

function setupMocks({
  userOrgs = { provider: "github", organizations: ["OpenHands", "AcmeCo"] },
  claims = [] as Array<{
    id: string;
    org_id: string;
    provider: string;
    git_organization: string;
    claimed_by: string;
    claimed_at: string;
  }>,
  isLoadingUserOrgs = false,
  isLoadingClaims = false,
} = {}) {
  mockUseUserGitOrganizations.mockReturnValue({
    data: userOrgs,
    isLoading: isLoadingUserOrgs,
  } as ReturnType<typeof useUserGitOrganizations>);

  mockUseGitClaims.mockReturnValue({
    data: claims,
    isLoading: isLoadingClaims,
  } as ReturnType<typeof useGitClaims>);
}

describe("useGitConversationRouting", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns empty orgs when user git orgs is undefined", () => {
    mockUseUserGitOrganizations.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useUserGitOrganizations>);
    mockUseGitClaims.mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useGitClaims>);

    const { result } = renderHook(() => useGitConversationRouting());

    expect(result.current.orgs).toEqual([]);
    expect(result.current.isLoading).toBe(true);
  });

  it("merges user git orgs with claims correctly", () => {
    setupMocks({
      claims: [
        {
          id: "claim-1",
          org_id: "org-1",
          provider: "github",
          git_organization: "openhands",
          claimed_by: "user-1",
          claimed_at: "2026-01-01T00:00:00",
        },
      ],
    });

    const { result } = renderHook(() => useGitConversationRouting());

    expect(result.current.orgs).toHaveLength(2);

    const claimedOrg = result.current.orgs.find(
      (o) => o.name === "OpenHands",
    );
    expect(claimedOrg).toMatchObject({
      id: "github:openhands",
      claimId: "claim-1",
      provider: "github",
      status: "claimed",
    });

    const unclaimedOrg = result.current.orgs.find(
      (o) => o.name === "AcmeCo",
    );
    expect(unclaimedOrg).toMatchObject({
      id: "github:acmeco",
      claimId: null,
      provider: "github",
      status: "unclaimed",
    });
  });

  it("handles case-insensitive matching between user orgs and claims", () => {
    setupMocks({
      userOrgs: {
        provider: "GitHub",
        organizations: ["All-Hands-AI"],
      },
      claims: [
        {
          id: "claim-1",
          org_id: "org-1",
          provider: "github",
          git_organization: "all-hands-ai",
          claimed_by: "user-1",
          claimed_at: "2026-01-01T00:00:00",
        },
      ],
    });

    const { result } = renderHook(() => useGitConversationRouting());

    expect(result.current.orgs[0]).toMatchObject({
      status: "claimed",
      claimId: "claim-1",
      name: "All-Hands-AI",
    });
  });

  it("returns unclaimed when no claims data is available", () => {
    setupMocks({ claims: [] });

    const { result } = renderHook(() => useGitConversationRouting());

    expect(result.current.orgs).toHaveLength(2);
    expect(result.current.orgs.every((o) => o.status === "unclaimed")).toBe(
      true,
    );
    expect(result.current.orgs.every((o) => o.claimId === null)).toBe(true);
  });

  it("sets pending claiming state when claimOrg is called", () => {
    setupMocks();

    const { result } = renderHook(() => useGitConversationRouting());

    act(() => {
      result.current.claimOrg("github:acmeco");
    });

    expect(mockClaimMutate).toHaveBeenCalledWith(
      { provider: "github", gitOrganization: "AcmeCo" },
      expect.objectContaining({ onSettled: expect.any(Function) }),
    );

    const org = result.current.orgs.find((o) => o.id === "github:acmeco");
    expect(org?.status).toBe("claiming");
  });

  it("sets pending disconnecting state when disconnectOrg is called", () => {
    setupMocks({
      claims: [
        {
          id: "claim-1",
          org_id: "org-1",
          provider: "github",
          git_organization: "openhands",
          claimed_by: "user-1",
          claimed_at: "2026-01-01T00:00:00",
        },
      ],
    });

    const { result } = renderHook(() => useGitConversationRouting());

    act(() => {
      result.current.disconnectOrg("github:openhands");
    });

    expect(mockDisconnectMutate).toHaveBeenCalledWith(
      { claimId: "claim-1" },
      expect.objectContaining({ onSettled: expect.any(Function) }),
    );

    const org = result.current.orgs.find((o) => o.id === "github:openhands");
    expect(org?.status).toBe("disconnecting");
  });

  it("clears pending state on settle", () => {
    setupMocks();

    mockClaimMutate.mockImplementation(
      (
        _args: unknown,
        options: { onSettled?: () => void },
      ) => {
        options?.onSettled?.();
      },
    );

    const { result } = renderHook(() => useGitConversationRouting());

    act(() => {
      result.current.claimOrg("github:acmeco");
    });

    const org = result.current.orgs.find((o) => o.id === "github:acmeco");
    expect(org?.status).toBe("unclaimed");
  });

  it("does not claim an already claimed org", () => {
    setupMocks({
      claims: [
        {
          id: "claim-1",
          org_id: "org-1",
          provider: "github",
          git_organization: "openhands",
          claimed_by: "user-1",
          claimed_at: "2026-01-01T00:00:00",
        },
      ],
    });

    const { result } = renderHook(() => useGitConversationRouting());

    act(() => {
      result.current.claimOrg("github:openhands");
    });

    expect(mockClaimMutate).not.toHaveBeenCalled();
  });

  it("does not disconnect an unclaimed org", () => {
    setupMocks();

    const { result } = renderHook(() => useGitConversationRouting());

    act(() => {
      result.current.disconnectOrg("github:acmeco");
    });

    expect(mockDisconnectMutate).not.toHaveBeenCalled();
  });
});
