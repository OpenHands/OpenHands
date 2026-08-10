import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { setStoredConversationMetadata } from "#/api/conversation-metadata-store";

const useAgentProfilesMock = vi.fn();
const useActiveConversationMock = vi.fn();
const activateProfileMutate = vi.fn();
const createConversationMutate = vi.fn();

vi.mock("#/hooks/query/use-agent-profiles", () => ({
  useAgentProfiles: () => useAgentProfilesMock(),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({ mutateAsync: createConversationMutate }),
  CREATE_CONVERSATION_MUTATION_KEY: ["create-conversation"],
}));

vi.mock("#/hooks/mutation/use-activate-agent-profile", () => ({
  useActivateAgentProfile: () => ({
    mutate: activateProfileMutate,
    isPending: false,
  }),
  ACTIVATE_AGENT_PROFILE_MUTATION_KEY: ["activate-agent-profile"],
}));

import { ChatInputProfileMenuContent } from "#/components/features/chat/components/chat-input-profile-picker";

const PROFILES = [
  { id: "id-default", name: "Default", agent_kind: "openhands" },
  { id: "id-codex", name: "Codex", agent_kind: "acp" },
];

describe("ChatInputProfileMenuContent", () => {
  const onClose = vi.fn();

  beforeEach(() => {
    useAgentProfilesMock.mockReset();
    useActiveConversationMock.mockReset();
    activateProfileMutate.mockReset();
    createConversationMutate.mockReset();
    // selectProfile chains .then off mutateAsync — keep a resolved default.
    createConversationMutate.mockResolvedValue({ conversation_id: "conv-x" });
    onClose.mockReset();
    localStorage.clear();

    useAgentProfilesMock.mockReturnValue({
      data: { profiles: PROFILES, active_agent_profile_id: "id-default" },
      isLoading: false,
    });
    useActiveConversationMock.mockReturnValue({
      data: undefined,
      isLoading: false,
    });
  });

  // The menu content renders <li> children (it lives inside the tools menu's
  // submenu <ContextMenu> in the app), so tests wrap it in a bare <ul>.
  const renderMenu = (navigation: object = { conversationId: null }) =>
    renderWithProviders(
      <ul>
        <ChatInputProfileMenuContent onClose={onClose} />
      </ul>,
      { navigation },
    );

  it("activates the picked profile on the home page", () => {
    renderMenu();
    fireEvent.click(
      screen.getByTestId("chat-input-agent-profile-option-Codex"),
    );

    expect(activateProfileMutate).toHaveBeenCalledWith("id-codex");
    expect(onClose).toHaveBeenCalled();
  });

  it("does not activate when the active profile is re-selected", () => {
    renderMenu();
    fireEvent.click(
      screen.getByTestId("chat-input-agent-profile-option-Default"),
    );

    expect(activateProfileMutate).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it("links to the AgentProfile library in settings", () => {
    const { container } = renderMenu();

    expect(
      container.ownerDocument.querySelector('a[href="/settings/agents"]'),
    ).not.toBeNull();
  });

  it("treats the launched profile as current inside a blank conversation", () => {
    // The active pointer is Codex, but this blank conversation was launched
    // from Default — re-selecting Default must be a no-op (no replacement
    // conversation, no activate).
    useAgentProfilesMock.mockReturnValue({
      data: { profiles: PROFILES, active_agent_profile_id: "id-codex" },
      isLoading: false,
    });
    useActiveConversationMock.mockReturnValue({
      data: {
        id: "conv-1",
        launched_agent_profile: {
          agent_profile_id: "id-default",
          revision: 2,
        },
      },
      isLoading: false,
    });

    renderMenu({ conversationId: "conv-1" });
    fireEvent.click(
      screen.getByTestId("chat-input-agent-profile-option-Default"),
    );

    expect(createConversationMutate).not.toHaveBeenCalled();
    expect(activateProfileMutate).not.toHaveBeenCalled();
  });

  it("starts a replacement conversation with the selected profile and workspace", async () => {
    const navigate = vi.fn();
    createConversationMutate.mockResolvedValue({ conversation_id: "conv-2" });
    useActiveConversationMock.mockReturnValue({
      data: {
        id: "conv-workspace",
        selected_repository: null,
        selected_workspace: "/workspace/alpha",
        launched_agent_profile: {
          agent_profile_id: "id-default",
          revision: 1,
        },
      },
      isLoading: false,
    });

    renderMenu({ conversationId: "conv-workspace", navigate });

    expect(screen.getByText("CHAT$START_NEW_WITH_PROFILE_HINT")).toBeVisible();
    fireEvent.click(
      screen.getByTestId("chat-input-agent-profile-option-Codex"),
    );

    expect(createConversationMutate).toHaveBeenCalledWith({
      agentProfileId: "id-codex",
      entryPoint: "blank_conversation_profile_picker",
      workingDir: "/workspace/alpha",
      workspaceMode: "local_repo",
    });
    expect(activateProfileMutate).not.toHaveBeenCalled();

    // Navigation rides the mutateAsync promise (survives the menu unmounting).
    await waitFor(() =>
      expect(navigate).toHaveBeenCalledWith("/conversations/conv-2"),
    );
  });

  it("preserves repository and plugin context when changing a blank conversation profile", () => {
    setStoredConversationMetadata("conv-repo", {
      selected_repository: "OpenHands/agent-canvas",
      selected_branch: "feature",
      git_provider: "github",
      plugins: [
        {
          source: "github:OpenHands/extensions",
          ref: "v1",
          repo_path: "plugins/weather",
        },
      ],
    });
    useActiveConversationMock.mockReturnValue({
      data: {
        id: "conv-repo",
        selected_repository: "OpenHands/agent-canvas",
        selected_branch: "feature",
        git_provider: "github",
        launched_agent_profile: {
          agent_profile_id: "id-default",
          revision: 1,
        },
      },
      isLoading: false,
    });

    renderMenu({ conversationId: "conv-repo" });
    fireEvent.click(
      screen.getByTestId("chat-input-agent-profile-option-Codex"),
    );

    expect(createConversationMutate).toHaveBeenCalledWith(
      expect.objectContaining({
        agentProfileId: "id-codex",
        repository: {
          name: "OpenHands/agent-canvas",
          gitProvider: "github",
          branch: "feature",
        },
        plugins: [
          {
            source: "github:OpenHands/extensions",
            ref: "v1",
            repo_path: "plugins/weather",
          },
        ],
      }),
    );
  });
});
