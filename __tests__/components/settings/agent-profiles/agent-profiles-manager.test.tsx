import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AgentProfilesManager } from "#/components/features/settings/agent-profiles/agent-profiles-manager";
import AgentProfilesService, {
  type AgentProfile,
  type AgentProfileSummary,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        SETTINGS$AVAILABLE_PROFILES: "Available Profiles",
        SETTINGS$ADD_AGENT_PROFILE: "Add agent profile",
        SETTINGS$PROFILES_LOAD_ERROR: "Failed to load profiles",
        SETTINGS$PROFILES_EMPTY: "No profiles saved yet",
        SETTINGS$PROFILE_ACTIVE: "Active",
        SETTINGS$PROFILE_DEFAULT: "Default",
        SETTINGS$PROFILE_MENU: "Profile menu",
        SETTINGS$PROFILE_EDIT: "Edit",
        SETTINGS$PROFILE_SET_ACTIVE: "Set as active",
        SETTINGS$PROFILE_DELETE_TITLE: "Delete Profile",
        SETTINGS$AGENT_TYPE_ACP: "ACP",
        SETTINGS$PROFILE_DELETE_CONFIRMATION: params?.name
          ? `Are you sure you want to delete "${params.name}"?`
          : "Are you sure you want to delete this profile?",
        SETTINGS$PROFILE_ACTIVATED: params?.name
          ? `Profile "${params.name}" activated`
          : "Profile activated",
        SETTINGS$PROFILE_DUPLICATED: params?.name
          ? `Profile "${params.name}" duplicated`
          : "Profile duplicated",
        BUTTON$DUPLICATE: "Duplicate",
        BUTTON$DELETE: "Delete",
        BUTTON$CANCEL: "Cancel",
        ERROR$GENERIC: "An error occurred",
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api");
vi.mock("#/utils/custom-toast-handlers");
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "b1", kind: "local" },
    orgId: null,
  }),
}));

// Gating hook: `true` by default (local / owner-admin); flipped per test to
// exercise the cloud view-only member path.
const canManage = vi.hoisted(() => ({ value: true }));
vi.mock("#/hooks/use-can-manage-org-profiles", () => ({
  useCanManageOrgProfiles: () => canManage.value,
}));

const mockProfiles: AgentProfileSummary[] = [
  {
    id: "id-oh",
    name: "my-openhands",
    agent_kind: "openhands",
    revision: 0,
    llm_profile_ref: "default",
    mcp_server_refs: null,
  },
  {
    id: "id-acp",
    name: "my-claude",
    agent_kind: "acp",
    revision: 0,
    llm_profile_ref: null,
    mcp_server_refs: null,
  },
];

describe("AgentProfilesManager", () => {
  let queryClient: QueryClient;

  const renderManager = (
    props: {
      onAddProfile?: () => void;
      onEditProfile?: (profile: AgentProfileSummary) => void;
    } = {},
  ) =>
    render(
      <QueryClientProvider client={queryClient}>
        <AgentProfilesManager {...props} />
      </QueryClientProvider>,
    );

  beforeEach(() => {
    canManage.value = true;
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  it("displays the section title and profiles", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_agent_profile_id: "id-oh",
    });

    renderManager();

    expect(screen.getByText("Available Profiles")).toBeInTheDocument();
    await screen.findByText("my-openhands");
    expect(screen.getByText("my-claude")).toBeInTheDocument();
    // OpenHands profiles show their referenced LLM profile; ACP shows the kind.
    expect(screen.getByText("default")).toBeInTheDocument();
    expect(screen.getByText("ACP")).toBeInTheDocument();
  });

  it("marks the active profile", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_agent_profile_id: "id-oh",
    });

    renderManager();

    const badge = await screen.findByTestId("agent-profile-active-badge");
    expect(badge).toHaveTextContent("Active");
  });

  it("keeps the default profile from being deleted", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [{ ...mockProfiles[0], id: "id-default", name: "default" }],
      active_agent_profile_id: "id-default",
    });

    renderManager();
    const user = userEvent.setup();
    await user.click(await screen.findByTestId("agent-profile-menu-trigger"));

    const remove = screen.getByTestId("agent-profile-delete");
    expect(remove).toBeDisabled();
    expect(remove).toHaveAttribute(
      "title",
      "SETTINGS$AGENT_PROFILE_DEFAULT_LOCKED",
    );
    expect(screen.getByTestId("agent-profile-duplicate")).toBeEnabled();
  });

  it("shows the Add button and fires onAddProfile", async () => {
    const onAddProfile = vi.fn();
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_agent_profile_id: null,
    });

    renderManager({ onAddProfile });

    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-agent-profile"));
    expect(onAddProfile).toHaveBeenCalledTimes(1);
  });

  it("hides mutate controls for view-only members (canManage false)", async () => {
    canManage.value = false;
    const onAddProfile = vi.fn();
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_agent_profile_id: "id-oh",
    });

    renderManager({ onAddProfile });

    // The list still renders, but there is no Add button and no row actions.
    await screen.findByText("my-openhands");
    expect(screen.queryByTestId("add-agent-profile")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-profile-menu-trigger"),
    ).not.toBeInTheDocument();
  });

  it("shows the empty state when there are no profiles", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_agent_profile_id: null,
    });

    renderManager();

    await screen.findByText("No profiles saved yet");
  });

  it("shows an error message when loading fails", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockRejectedValue(
      new Error("Network error"),
    );

    renderManager();

    await screen.findByText("Failed to load profiles");
  });

  it("calls onEditProfile from the row menu", async () => {
    const onEditProfile = vi.fn();
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_agent_profile_id: "id-oh",
    });

    renderManager({ onEditProfile });

    await screen.findByText("my-openhands");
    const user = userEvent.setup();
    const triggers = screen.getAllByTestId("agent-profile-menu-trigger");
    await user.click(triggers[0]);
    await user.click(screen.getByText("Edit"));

    expect(onEditProfile).toHaveBeenCalledWith(mockProfiles[0]);
  });

  it("duplicates an OpenHands agent's full configuration with a unique name and new identity", async () => {
    const source = {
      id: "id-oh",
      name: "my-openhands",
      revision: 7,
      schema_version: 1,
      agent_kind: "openhands",
      llm_profile_ref: "fast-llm",
      mcp_server_refs: ["github"],
      secret_refs: ["GITHUB_TOKEN"],
      disabled_skills: ["unused-skill"],
      tools: [{ name: "terminal", params: {} }],
      persona: "Careful reviewer",
    } as unknown as AgentProfile;
    let summaries = [
      ...mockProfiles,
      { ...mockProfiles[0], id: "id-copy", name: "my-openhands-copy" },
    ];
    vi.mocked(AgentProfilesService.listProfiles).mockImplementation(
      async () => ({ profiles: summaries, active_agent_profile_id: "id-oh" }),
    );
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: source.name,
      profile: source,
    });
    vi.mocked(AgentProfilesService.saveProfile).mockImplementation(
      async (name) => {
        summaries = [...summaries, { ...mockProfiles[0], id: "new-id", name }];
        return { name, message: "ok" } as never;
      },
    );

    renderManager();
    await screen.findByText("my-openhands");
    const user = userEvent.setup();
    await user.click(screen.getAllByTestId("agent-profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("agent-profile-duplicate"));

    await waitFor(() =>
      expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
        "my-openhands-copy-1",
        expect.objectContaining({
          agent_kind: "openhands",
          llm_profile_ref: "fast-llm",
          mcp_server_refs: ["github"],
          secret_refs: ["GITHUB_TOKEN"],
          disabled_skills: ["unused-skill"],
          tools: [{ name: "terminal", params: {} }],
          persona: "Careful reviewer",
        }),
      ),
    );
    expect(AgentProfilesService.getProfile).toHaveBeenCalledWith(
      "my-openhands",
      "encrypted",
    );
    const cloned = vi.mocked(AgentProfilesService.saveProfile).mock
      .lastCall?.[1];
    expect(cloned).not.toHaveProperty("id");
    expect(cloned).not.toHaveProperty("name");
    expect(cloned).not.toHaveProperty("revision");
    expect(cloned).not.toHaveProperty("schema_version");
    expect(await screen.findByText("my-openhands-copy-1")).toBeInTheDocument();
    expect(displaySuccessToast).toHaveBeenCalledWith(
      'Profile "my-openhands-copy-1" duplicated',
    );
    expect(AgentProfilesService.activateProfile).not.toHaveBeenCalled();
  });

  it("duplicates ACP agents without injecting OpenHands fields and truncates long names", async () => {
    const longName = "a".repeat(62);
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [{ ...mockProfiles[1], name: longName }],
      active_agent_profile_id: "id-acp",
    });
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      name: longName,
      profile: {
        id: "id-acp",
        name: longName,
        revision: 2,
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: "claude",
        acp_args: ["--verbose"],
        secret_refs: ["ANTHROPIC_API_KEY"],
        mcp_server_refs: ["github"],
      } as AgentProfile,
    });
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: `${"a".repeat(59)}-copy`,
      message: "ok",
    } as never);

    renderManager();
    const user = userEvent.setup();
    await user.click(await screen.findByTestId("agent-profile-menu-trigger"));
    await user.click(screen.getByTestId("agent-profile-duplicate"));

    await waitFor(() =>
      expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
        `${"a".repeat(59)}-copy`,
        expect.objectContaining({
          agent_kind: "acp",
          acp_command: "claude",
          acp_args: ["--verbose"],
          secret_refs: ["ANTHROPIC_API_KEY"],
        }),
      ),
    );
    expect(
      vi.mocked(AgentProfilesService.saveProfile).mock.lastCall?.[1],
    ).not.toHaveProperty("llm_profile_ref");
  });

  it("does not create a duplicate if loading the agent fails", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_agent_profile_id: "id-oh",
    });
    vi.mocked(AgentProfilesService.getProfile).mockRejectedValue(
      new Error("Not found"),
    );

    renderManager();
    await screen.findByText("my-openhands");
    const user = userEvent.setup();
    await user.click(screen.getAllByTestId("agent-profile-menu-trigger")[0]);
    await user.click(screen.getByTestId("agent-profile-duplicate"));

    await waitFor(() => expect(displayErrorToast).toHaveBeenCalled());
    expect(AgentProfilesService.saveProfile).not.toHaveBeenCalled();
  });

  it("activates a profile by id from the row menu", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_agent_profile_id: "id-oh",
    });
    vi.mocked(AgentProfilesService.activateProfile).mockResolvedValue({
      id: "id-acp",
      message: "ok",
      agent_settings_applied: false,
    });

    renderManager();

    await screen.findByText("my-claude");
    const user = userEvent.setup();
    // Second row (my-claude) is not active, so Set active is enabled.
    const triggers = screen.getAllByTestId("agent-profile-menu-trigger");
    await user.click(triggers[1]);
    await user.click(screen.getByText("Set as active"));

    expect(AgentProfilesService.activateProfile).toHaveBeenCalledWith("id-acp");
  });

  it("opens the delete modal from the row menu", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: mockProfiles,
      active_agent_profile_id: "id-oh",
    });

    renderManager();

    await screen.findByText("my-claude");
    const user = userEvent.setup();
    const triggers = screen.getAllByTestId("agent-profile-menu-trigger");
    await user.click(triggers[1]);
    await user.click(screen.getByText("Delete"));

    expect(
      screen.getByText('Are you sure you want to delete "my-claude"?'),
    ).toBeInTheDocument();
  });
});
