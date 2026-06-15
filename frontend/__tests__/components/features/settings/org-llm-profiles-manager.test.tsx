import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { OrgLlmProfilesManager } from "#/components/features/settings/org-llm-profiles-manager";

type ProfilesList = {
  profiles: LlmProfileSummary[];
  active_profile: string | null;
};

const profilesState: {
  data: ProfilesList | undefined;
  isLoading: boolean;
  error: Error | null;
} = { data: undefined, isLoading: false, error: null };

const activateMock = vi.fn();
const deleteMock = vi.fn();
const renameMock = vi.fn();

vi.mock("#/hooks/query/use-org-llm-profiles", () => ({
  useOrgLlmProfiles: () => profilesState,
}));

vi.mock("#/hooks/mutation/use-org-llm-profile-mutations", () => ({
  useActivateOrgLlmProfile: () => ({
    mutateAsync: activateMock,
    isPending: false,
  }),
  useDeleteOrgLlmProfile: () => ({
    mutateAsync: deleteMock,
    isPending: false,
  }),
  useRenameOrgLlmProfile: () => ({
    mutateAsync: renameMock,
    isPending: false,
  }),
}));

const sampleProfiles: ProfilesList = {
  profiles: [
    {
      name: "sonnet",
      model: "openhands/claude-sonnet-4-5-20250929",
      base_url: null,
      api_key_set: true,
    },
    {
      name: "opus",
      model: "openhands/claude-opus-4-7",
      base_url: null,
      api_key_set: true,
    },
  ],
  active_profile: "sonnet",
};

function renderManager({
  canManage,
  onAddProfile,
  onEditProfile,
}: {
  canManage?: boolean;
  onAddProfile?: () => void;
  onEditProfile?: (profile: LlmProfileSummary) => void;
} = {}) {
  return render(
    <OrgLlmProfilesManager
      orgId="org-1"
      canManage={canManage}
      onAddProfile={onAddProfile}
      onEditProfile={onEditProfile}
    />,
  );
}

beforeEach(() => {
  profilesState.data = sampleProfiles;
  profilesState.isLoading = false;
  profilesState.error = null;
  activateMock.mockReset().mockResolvedValue(undefined);
  deleteMock.mockReset().mockResolvedValue(undefined);
  renameMock.mockReset().mockResolvedValue(undefined);
});

describe("OrgLlmProfilesManager", () => {
  it("renders Add LLM Profile when management is enabled", async () => {
    const onAddProfile = vi.fn();
    renderManager({ canManage: true, onAddProfile });
    const user = userEvent.setup();

    await user.click(screen.getByTestId("add-llm-profile"));

    expect(onAddProfile).toHaveBeenCalledTimes(1);
  });

  it("renders profiles without management controls when management is disabled", () => {
    renderManager({
      canManage: false,
      onAddProfile: vi.fn(),
      onEditProfile: vi.fn(),
    });

    expect(screen.getByText("sonnet")).toBeInTheDocument();
    expect(screen.getByText("opus")).toBeInTheDocument();
    expect(screen.queryByTestId("add-llm-profile")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("profile-menu-trigger"),
    ).not.toBeInTheDocument();
  });
});
