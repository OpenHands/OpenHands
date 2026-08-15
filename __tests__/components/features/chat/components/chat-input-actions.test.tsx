import React from "react";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";

const useActiveConversationMock = vi.fn<
  () => {
    data:
      | {
          conversation_id: string;
          agent_kind?: "openhands" | "acp";
          llm_model: string | null;
        }
      | undefined;
  }
>(() => ({ data: undefined }));

vi.mock("#/components/features/controls/agent-status", () => ({
  AgentStatus: () => <div data-testid="agent-status-stub" />,
}));

vi.mock("#/components/features/chat/change-agent-button", () => ({
  ChangeAgentButton: () => <div data-testid="change-agent-button-stub" />,
}));

vi.mock(
  "#/components/features/chat/components/chat-input-profile-picker",
  () => ({
    ChatInputProfileMenuContent: () => (
      <div data-testid="agent-profile-menu-stub" />
    ),
  }),
);

vi.mock(
  "#/components/features/chat/components/chat-input-llm-profile-picker",
  () => ({
    ChatInputLlmProfilePicker: () => (
      <div data-testid="llm-profile-picker-stub" />
    ),
    ChatInputLlmProfileMenuContent: () => (
      <div data-testid="llm-profile-menu-stub" />
    ),
  }),
);

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

// Service-level mock (not the hook): drives the Switch-agent-profile gate,
// which reads the real useAgentProfiles query.
const listAgentProfilesMock = vi.fn();
vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  default: {
    listProfiles: (...args: unknown[]) => listAgentProfilesMock(...args),
    getProfile: vi.fn(),
    saveProfile: vi.fn(),
    activateProfile: vi.fn(),
  },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  pauseConversation: vi.fn(),
  resumeConversation: vi.fn(),
  askAgent: vi.fn(),
  updateConversationExecutionStatusInCache: vi.fn(),
  invalidateConversationQueries: vi.fn(),
}));

// eslint-disable-next-line import/first
import { ChatInputActions } from "#/components/features/chat/components/chat-input-actions";

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

describe("ChatInputActions", () => {
  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    useActiveConversationMock.mockReset();
    useActiveConversationMock.mockReturnValue({ data: undefined });
  });

  it("renders the LLM-profile picker on the home page (local)", () => {
    useActiveConversationMock.mockReturnValue({ data: undefined });

    renderWithProviders(<ChatInputActions disabled={false} />, {
      navigation: { conversationId: null },
    });

    // The pill is always an LLM selector (OSS-5735) — agent-profile switching
    // lives in the "+" tools menu instead.
    expect(screen.getByTestId("llm-profile-picker-stub")).toBeInTheDocument();
    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("renders the LLM-profile picker on the home page (cloud)", () => {
    // Cloud used to fall back to a read-only model chip before a conversation
    // started, surfacing the raw model path instead of the profile name (#1932
    // review). The pill names the profile on every backend now.
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    useActiveConversationMock.mockReturnValue({ data: undefined });

    renderWithProviders(<ChatInputActions disabled={false} />, {
      navigation: { conversationId: null },
    });

    expect(screen.getByTestId("llm-profile-picker-stub")).toBeInTheDocument();
    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("renders the LLM-profile picker inside a blank local OpenHands conversation", () => {
    useActiveConversationMock.mockReturnValue({
      data: { conversation_id: "test-conversation-id", llm_model: null },
    });

    renderWithProviders(
      <ChatInputActions disabled={false} hasStartedConversation={false} />,
      {
        navigation: { conversationId: "test-conversation-id" },
      },
    );

    expect(screen.getByTestId("llm-profile-picker-stub")).toBeInTheDocument();
    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("renders the LLM-profile switcher inside a started local OpenHands conversation", () => {
    useActiveConversationMock.mockReturnValue({
      data: { conversation_id: "test-conversation-id", llm_model: "gpt-4o" },
    });

    renderWithProviders(
      <ChatInputActions disabled={false} hasStartedConversation />,
      {
        navigation: { conversationId: "test-conversation-id" },
      },
    );

    // In a conversation the user live-switches the LLM profile.
    expect(screen.getByTestId("llm-profile-picker-stub")).toBeInTheDocument();
    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("renders the model switcher inside a local ACP conversation", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        llm_model: "claude-sonnet-4-6",
      },
    });

    renderWithProviders(
      <ChatInputActions disabled={false} hasStartedConversation />,
      {
        navigation: { conversationId: "test-conversation-id" },
      },
    );

    // ACP in a conversation live-switches the running model via ChatInputModel.
    expect(screen.getByTestId("chat-input-llm-model")).toBeInTheDocument();
    expect(
      screen.queryByTestId("llm-profile-picker-stub"),
    ).not.toBeInTheDocument();
  });

  it("renders the active conversation model in a cloud ACP conversation", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        llm_model: "gpt-4o",
      },
    });

    renderWithProviders(
      <ActiveBackendProvider>
        <ChatInputActions disabled={false} hasStartedConversation />
      </ActiveBackendProvider>,
    );

    expect(screen.getByTestId("chat-input-llm-model")).toHaveTextContent(
      "gpt-4o",
    );
  });

  it("omits the model label on cloud when the active ACP conversation has no llm_model", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        llm_model: null,
      },
    });

    renderWithProviders(
      <ActiveBackendProvider>
        <ChatInputActions disabled={false} hasStartedConversation />
      </ActiveBackendProvider>,
    );

    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("renders the LLM-profile switcher inside a cloud OpenHands conversation", () => {
    // /switch_profile is a real endpoint on both backends (cloud proxies
    // POST /api/v1/app-conversations/{id}/switch_profile) — cloud OpenHands
    // conversations get the same live-switch picker as local (#1571 review).
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    useActiveConversationMock.mockReturnValue({
      data: { conversation_id: "test-conversation-id", llm_model: "gpt-4o" },
    });

    renderWithProviders(
      <ActiveBackendProvider>
        <ChatInputActions disabled={false} hasStartedConversation />
      </ActiveBackendProvider>,
    );

    expect(screen.getByTestId("llm-profile-picker-stub")).toBeInTheDocument();
    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("hides the Change Agent button on a local backend", () => {
    renderWithProviders(<ChatInputActions disabled={false} />);

    expect(
      screen.queryByTestId("change-agent-button-stub"),
    ).not.toBeInTheDocument();
  });

  it("shows the Change Agent button on a cloud backend", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    renderWithProviders(
      <ActiveBackendProvider>
        <ChatInputActions disabled={false} />
      </ActiveBackendProvider>,
    );

    expect(screen.getByTestId("change-agent-button-stub")).toBeInTheDocument();
  });

  it("shows the Change Agent button on the home page on a cloud backend", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    renderWithProviders(
      <ActiveBackendProvider>
        <ChatInputActions disabled={false} />
      </ActiveBackendProvider>,
      { navigation: { conversationId: null } },
    );

    expect(screen.getByTestId("change-agent-button-stub")).toBeInTheDocument();
  });
});

describe("ChatInputActions — Switch agent profile gate (OSS-5735)", () => {
  beforeEach(() => {
    useActiveConversationMock.mockReset();
    useActiveConversationMock.mockReturnValue({ data: undefined });
    listAgentProfilesMock.mockReset();
    listAgentProfilesMock.mockResolvedValue({
      profiles: [{ id: "p1", name: "default", agent_kind: "openhands" }],
      active_agent_profile_id: "p1",
    });
  });

  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  const openPlusMenu = () => {
    fireEvent.click(screen.getByTestId("chat-plus-button"));
  };

  it("offers Switch agent profile in the + menu on the home page when profiles exist", async () => {
    renderWithProviders(<ChatInputActions disabled={false} />, {
      navigation: { conversationId: null },
    });

    openPlusMenu();

    expect(
      await screen.findByTestId("switch-agent-profile-button"),
    ).toBeInTheDocument();
  });

  it("offers Switch agent profile in a blank (unstarted) conversation", async () => {
    useActiveConversationMock.mockReturnValue({
      data: { conversation_id: "conv-blank", llm_model: null },
    });

    renderWithProviders(
      <ChatInputActions disabled={false} hasStartedConversation={false} />,
      { navigation: { conversationId: "conv-blank" } },
    );

    openPlusMenu();

    expect(
      await screen.findByTestId("switch-agent-profile-button"),
    ).toBeInTheDocument();
  });

  it("does not offer Switch agent profile once the conversation has started", async () => {
    useActiveConversationMock.mockReturnValue({
      data: { conversation_id: "conv-started", llm_model: "gpt-4o" },
    });

    renderWithProviders(
      <ChatInputActions disabled={false} hasStartedConversation />,
      { navigation: { conversationId: "conv-started" } },
    );

    openPlusMenu();

    expect(await screen.findByTestId("tools-context-menu")).toBeInTheDocument();
    expect(
      screen.queryByTestId("switch-agent-profile-button"),
    ).not.toBeInTheDocument();
  });

  it("does not offer Switch agent profile when the backend has no profiles", async () => {
    listAgentProfilesMock.mockResolvedValue({
      profiles: [],
      active_agent_profile_id: null,
    });

    renderWithProviders(<ChatInputActions disabled={false} />, {
      navigation: { conversationId: null },
    });

    openPlusMenu();

    await waitFor(() => expect(listAgentProfilesMock).toHaveBeenCalled());
    expect(
      screen.queryByTestId("switch-agent-profile-button"),
    ).not.toBeInTheDocument();
  });

  it("does not offer Switch agent profile while a cloud start task is provisioning", async () => {
    renderWithProviders(
      <ChatInputActions disabled={false} hasStartedConversation={false} />,
      { navigation: { conversationId: "task-start-1" } },
    );

    openPlusMenu();

    await waitFor(() => expect(listAgentProfilesMock).toHaveBeenCalled());
    expect(
      screen.queryByTestId("switch-agent-profile-button"),
    ).not.toBeInTheDocument();
  });
});
