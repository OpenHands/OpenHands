import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { LlmProfilesManager } from "#/components/features/settings/llm-profiles-manager";

// Same shape as the service's listProfiles response — redeclared here
// instead of imported because the source type is module-private.
type ProfilesList = {
  profiles: LlmProfileSummary[];
  active_profile: string | null;
};

// State driven by the tests: the mocked useLlmProfiles hook reads from it.
const profilesState: {
  data: ProfilesList | undefined;
  isLoading: boolean;
  error: Error | null;
} = { data: undefined, isLoading: false, error: null };

const saveMock = vi.fn();
const deleteMock = vi.fn();
const activateMock = vi.fn();
const renameMock = vi.fn();

vi.mock("#/hooks/query/use-llm-profiles", async () => {
  const actual = await vi.importActual<
    typeof import("#/hooks/query/use-llm-profiles")
  >("#/hooks/query/use-llm-profiles");
  return {
    ...actual,
    useLlmProfiles: () => profilesState,
  };
});

vi.mock("#/hooks/mutation/use-save-llm-profile", () => ({
  useSaveLlmProfile: () => ({ mutateAsync: saveMock, isPending: false }),
}));
vi.mock("#/hooks/mutation/use-delete-llm-profile", () => ({
  useDeleteLlmProfile: () => ({ mutateAsync: deleteMock, isPending: false }),
}));
vi.mock("#/hooks/mutation/use-activate-llm-profile", () => ({
  useActivateLlmProfile: () => ({
    mutateAsync: activateMock,
    isPending: false,
  }),
}));
vi.mock("#/hooks/mutation/use-rename-llm-profile", () => ({
  useRenameLlmProfile: () => ({ mutateAsync: renameMock, isPending: false }),
}));

const toastMocks = vi.hoisted(() => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));
vi.mock("#/utils/custom-toast-handlers", () => toastMocks);

function renderManager({
  onAddProfile,
  onEditProfile,
}: {
  onAddProfile?: () => void;
  onEditProfile?: (profile: LlmProfileSummary) => void;
} = {}) {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <LlmProfilesManager
        onAddProfile={onAddProfile}
        onEditProfile={onEditProfile}
      />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  profilesState.data = undefined;
  profilesState.isLoading = false;
  profilesState.error = null;
  saveMock.mockReset().mockResolvedValue(undefined);
  deleteMock.mockReset().mockResolvedValue(undefined);
  activateMock.mockReset().mockResolvedValue(undefined);
  renameMock.mockReset().mockResolvedValue(undefined);
  toastMocks.displayErrorToast.mockReset();
  toastMocks.displaySuccessToast.mockReset();
});

// A realistic two-profile response used by several tests.
const sampleProfiles: ProfilesList = {
  profiles: [
    {
      name: "openai_gpt-4o",
      model: "openai/gpt-4o",
      base_url: null,
      api_key_set: true,
    },
    {
      name: "anthropic_claude",
      model: "anthropic/claude-3-5-sonnet",
      base_url: "https://api.anthropic.com",
      api_key_set: false,
    },
  ],
  active_profile: "openai_gpt-4o",
};

describe("LlmProfilesManager", () => {
  it("shows the Available Profiles heading", () => {
    profilesState.data = sampleProfiles;
    renderManager();

    expect(
      screen.getByText("SETTINGS$AVAILABLE_PROFILES"),
    ).toBeInTheDocument();
  });

  it("renders the Add LLM Profile button when an onAddProfile callback is supplied", async () => {
    profilesState.data = sampleProfiles;
    const onAddProfile = vi.fn();
    renderManager({ onAddProfile });
    const user = userEvent.setup();

    const button = screen.getByTestId("add-llm-profile");
    expect(button).toBeInTheDocument();
    await user.click(button);
    expect(onAddProfile).toHaveBeenCalledTimes(1);
  });

  it("omits the Add LLM Profile button when no callback is provided", () => {
    profilesState.data = sampleProfiles;
    renderManager();

    expect(screen.queryByTestId("add-llm-profile")).not.toBeInTheDocument();
  });

  it("shows the empty-state message when no profiles are saved", () => {
    profilesState.data = { profiles: [], active_profile: null };
    renderManager();

    expect(screen.getByText("SETTINGS$PROFILES_EMPTY")).toBeInTheDocument();
    expect(screen.queryByTestId("profile-row")).not.toBeInTheDocument();
  });

  it("shows the loading spinner while fetching", () => {
    profilesState.isLoading = true;
    renderManager();

    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });

  it("shows the error message when the query fails", () => {
    profilesState.error = new Error("boom");
    renderManager();

    expect(
      screen.getByText("SETTINGS$PROFILES_LOAD_ERROR"),
    ).toBeInTheDocument();
  });

  it("renders one row per profile and marks the active one with the Active badge", () => {
    profilesState.data = sampleProfiles;
    renderManager();

    const rows = screen.getAllByTestId("profile-row");
    expect(rows).toHaveLength(2);

    const activeRow = rows[0];
    expect(activeRow).toHaveTextContent("openai_gpt-4o");
    expect(activeRow).toHaveTextContent("openai/gpt-4o");
    expect(activeRow).toHaveTextContent("SETTINGS$PROFILE_ACTIVE_BADGE");

    const otherRow = rows[1];
    expect(otherRow).toHaveTextContent("anthropic_claude");
    expect(otherRow).not.toHaveTextContent("SETTINGS$PROFILE_ACTIVE_BADGE");
  });

  it("fires onEditProfile on Edit without activating the profile (inactive row)", async () => {
    profilesState.data = sampleProfiles;
    const onEditProfile = vi.fn();
    renderManager({ onEditProfile });
    const user = userEvent.setup();

    // Open the menu on the inactive (second) row.
    await user.click(screen.getAllByTestId("profile-menu-trigger")[1]);
    await user.click(screen.getByTestId("profile-edit"));

    // Edit must not activate the profile — activation is reserved for the
    // explicit "Set as active" action and the post-save flow.
    expect(activateMock).not.toHaveBeenCalled();
    expect(onEditProfile).toHaveBeenCalledTimes(1);
    expect(onEditProfile).toHaveBeenCalledWith(sampleProfiles.profiles[1]);
  });

  it("does not activate the profile when Edit is clicked on the active row", async () => {
    profilesState.data = sampleProfiles;
    const onEditProfile = vi.fn();
    renderManager({ onEditProfile });
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("profile-edit"));

    expect(activateMock).not.toHaveBeenCalled();
    expect(onEditProfile).toHaveBeenCalledWith(sampleProfiles.profiles[0]);
  });

  it("activates the profile when Set as active is clicked", async () => {
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[1]);
    await user.click(screen.getByTestId("profile-set-active"));

    expect(activateMock).toHaveBeenCalledTimes(1);
    expect(activateMock).toHaveBeenCalledWith("anthropic_claude");
  });

  it("disables Set as active on the already-active row", async () => {
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
    expect(screen.getByTestId("profile-set-active")).toBeDisabled();
  });

  it("opens the rename modal prefilled with the current name", async () => {
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("profile-rename"));

    const input = screen.getByTestId(
      "rename-profile-input",
    ) as HTMLInputElement;
    expect(input.value).toBe("openai_gpt-4o");
  });

  it("calls renameProfile with the trimmed new name", async () => {
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("profile-rename"));

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, "  my-favourite  ");
    await user.click(screen.getByTestId("rename-profile-submit"));

    expect(renameMock).toHaveBeenCalledTimes(1);
    expect(renameMock).toHaveBeenCalledWith({
      name: "openai_gpt-4o",
      newName: "my-favourite",
    });
  });

  it("rejects invalid rename names and does not call the mutation", async () => {
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("profile-rename"));

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, "has space");
    expect(screen.getByTestId("rename-profile-submit")).toBeDisabled();
    expect(renameMock).not.toHaveBeenCalled();
  });

  it("opens the delete modal and calls deleteProfile on confirmation", async () => {
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[1]);
    await user.click(screen.getByTestId("profile-delete"));

    expect(screen.getByTestId("delete-profile-confirm")).toBeInTheDocument();
    await user.click(screen.getByTestId("delete-profile-confirm"));

    expect(deleteMock).toHaveBeenCalledTimes(1);
    expect(deleteMock).toHaveBeenCalledWith("anthropic_claude");
  });

  it("closes the delete modal and skips the mutation when Cancel is clicked", async () => {
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[1]);
    await user.click(screen.getByTestId("profile-delete"));
    await user.click(screen.getByRole("button", { name: "BUTTON$CANCEL" }));

    expect(deleteMock).not.toHaveBeenCalled();
    expect(
      screen.queryByTestId("delete-profile-confirm"),
    ).not.toBeInTheDocument();
  });

  it("surfaces the server detail on a delete failure and keeps the modal open", async () => {
    deleteMock.mockRejectedValue({
      response: { data: { detail: "Profile 'x' not found" } },
    });
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[1]);
    await user.click(screen.getByTestId("profile-delete"));
    await user.click(screen.getByTestId("delete-profile-confirm"));

    expect(toastMocks.displayErrorToast).toHaveBeenCalledWith(
      "Profile 'x' not found",
    );
    expect(screen.getByTestId("delete-profile-confirm")).toBeInTheDocument();
  });

  it("falls back to a generic message when an activate failure has no detail", async () => {
    activateMock.mockRejectedValue({ message: "Network Error" });
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[1]);
    await user.click(screen.getByTestId("profile-set-active"));

    expect(toastMocks.displayErrorToast).toHaveBeenCalledWith("Network Error");
  });

  it("surfaces the server detail on a rename collision and keeps the modal open", async () => {
    renameMock.mockRejectedValue({
      response: { data: { detail: "Profile 'b' already exists" } },
    });
    profilesState.data = sampleProfiles;
    renderManager();
    const user = userEvent.setup();

    await user.click(screen.getAllByTestId("profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("profile-rename"));

    const input = screen.getByTestId("rename-profile-input");
    await user.clear(input);
    await user.type(input, "anthropic_claude");
    await user.click(screen.getByTestId("rename-profile-submit"));

    expect(toastMocks.displayErrorToast).toHaveBeenCalledWith(
      "Profile 'b' already exists",
    );
    expect(screen.getByTestId("rename-profile-submit")).toBeInTheDocument();
  });
});
