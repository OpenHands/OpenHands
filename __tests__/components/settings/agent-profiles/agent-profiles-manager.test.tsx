import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AgentProfilesManager } from "#/components/features/settings/agent-profiles/agent-profiles-manager";
import AgentProfilesService, {
  type AgentProfileSummary,
} from "#/api/agent-profiles-service/agent-profiles-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        SETTINGS$AVAILABLE_PROFILES: "Available Profiles",
        SETTINGS$ADD_AGENT_PROFILE: "Add agent profile",
        SETTINGS$PROFILES_LOAD_ERROR: "Failed to load profiles",
        SETTINGS$PROFILES_EMPTY: "No profiles saved yet",
        SETTINGS$PROFILE_ACTIVE: "Active",
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
