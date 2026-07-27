import { fireEvent, screen } from "@testing-library/react";
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
  useCreateConversation: () => ({ mutate: createConversationMutate }),
  CREATE_CONVERSATION_MUTATION_KEY: ["create-conversation"],
}));

vi.mock("#/hooks/mutation/use-activate-agent-profile", () => ({
  useActivateAgentProfile: () => ({
    mutate: activateProfileMutate,
    isPending: false,
  }),
  ACTIVATE_AGENT_PROFILE_MUTATION_KEY: ["activate-agent-profile"],
}));

import { ChatInputProfilePicker } from "#/components/features/chat/components/chat-input-profile-picker";

const PROFILES = [
  { id: "id-default", name: "Default", agent_kind: "openhands" },
  { id: "id-codex", name: "Codex", agent_kind: "acp" },
];

describe("ChatInputProfilePicker", () => {
  beforeEach(() => {
    useAgentProfilesMock.mockReset();
    useActiveConversationMock.mockReset();
    activateProfileMutate.mockReset();
    createConversationMutate.mockReset();
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

  const renderHomePicker = () =>
    renderWithProviders(<ChatInputProfilePicker />, {
      navigation: { conversationId: null },
    });

  it("renders nothing when there are no profiles", () => {
    useAgentProfilesMock.mockReturnValue({
      data: { profiles: [], active_agent_profile_id: null },
      isLoading: false,
    });

    const { container } = renderHomePicker();
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing while a cloud start task is provisioning", () => {
    const { container } = renderWithProviders(<ChatInputProfilePicker />, {
      navigation: { conversationId: "task-start-1" },
    });

    expect(container).toBeEmptyDOMElement();
  });

  it("labels the button with the active profile", () => {
    renderHomePicker();
    expect(screen.getByTestId("chat-input-agent-profile")).toHaveTextContent(
      "Default",
    );
  });

  it("activates the picked profile", () => {
    renderHomePicker();
    fireEvent.click(screen.getByTestId("chat-input-agent-profile"));
    fireEvent.click(
      screen.getByTestId("chat-input-agent-profile-option-Codex"),
    );

    expect(activateProfileMutate).toHaveBeenCalledWith("id-codex");
  });

  it("does not activate when the active profile is re-selected", () => {
    renderHomePicker();
    fireEvent.click(screen.getByTestId("chat-input-agent-profile"));
    fireEvent.click(
      screen.getByTestId("chat-input-agent-profile-option-Default"),
    );

    expect(activateProfileMutate).not.toHaveBeenCalled();
  });

  it("links to the AgentProfile library in settings", () => {
    const { container } = renderHomePicker();
    fireEvent.click(screen.getByTestId("chat-input-agent-profile"));

    expect(
      container.ownerDocument.querySelector('a[href="/settings/agents"]'),
    ).not.toBeNull();
  });

  it("shows the launched profile inside a blank conversation", () => {
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

    renderWithProviders(<ChatInputProfilePicker />, {
      navigation: { conversationId: "conv-1" },
    });

    expect(screen.getByTestId("chat-input-agent-profile")).toHaveTextContent(
      "Default",
    );
  });

  it("starts a replacement conversation with the selected profile and workspace", () => {
    const navigate = vi.fn();
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

    renderWithProviders(<ChatInputProfilePicker />, {
      navigation: { conversationId: "conv-workspace", navigate },
    });
    fireEvent.click(screen.getByTestId("chat-input-agent-profile"));

    expect(screen.getByText("CHAT$START_NEW_WITH_PROFILE_HINT")).toBeVisible();
    fireEvent.click(
      screen.getByTestId("chat-input-agent-profile-option-Codex"),
    );

    expect(createConversationMutate).toHaveBeenCalledWith(
      {
        agentProfileId: "id-codex",
        entryPoint: "blank_conversation_profile_picker",
        workingDir: "/workspace/alpha",
        workspaceMode: "local_repo",
      },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
    expect(activateProfileMutate).not.toHaveBeenCalled();

    const onSuccess = createConversationMutate.mock.calls[0]?.[1]?.onSuccess;
    onSuccess({ conversation_id: "conv-2" });
    expect(navigate).toHaveBeenCalledWith("/conversations/conv-2");
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

    renderWithProviders(<ChatInputProfilePicker />, {
      navigation: { conversationId: "conv-repo" },
    });
    fireEvent.click(screen.getByTestId("chat-input-agent-profile"));
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
      expect.any(Object),
    );
  });
});
