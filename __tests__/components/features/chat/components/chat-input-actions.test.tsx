import React from "react";
import { screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
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
    ChatInputProfilePicker: () => (
      <div data-testid="agent-profile-picker-stub" />
    ),
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

  it("renders the AgentProfile picker on the home page (local)", () => {
    useActiveConversationMock.mockReturnValue({ data: undefined });

    renderWithProviders(<ChatInputActions disabled={false} />, {
      navigation: { conversationId: null },
    });

    // Home keeps the start-new/activate AgentProfile picker (#3727).
    expect(screen.getByTestId("agent-profile-picker-stub")).toBeInTheDocument();
    expect(
      screen.queryByTestId("llm-profile-picker-stub"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("renders the AgentProfile picker inside a blank local OpenHands conversation", () => {
    useActiveConversationMock.mockReturnValue({
      data: { conversation_id: "test-conversation-id", llm_model: null },
    });

    renderWithProviders(
      <ChatInputActions disabled={false} hasStartedConversation={false} />,
      {
        navigation: { conversationId: "test-conversation-id" },
      },
    );

    expect(screen.getByTestId("agent-profile-picker-stub")).toBeInTheDocument();
    expect(
      screen.queryByTestId("llm-profile-picker-stub"),
    ).not.toBeInTheDocument();
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

    // In a conversation the user live-switches the LLM profile, not start-new.
    expect(screen.getByTestId("llm-profile-picker-stub")).toBeInTheDocument();
    expect(
      screen.queryByTestId("agent-profile-picker-stub"),
    ).not.toBeInTheDocument();
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
      screen.queryByTestId("agent-profile-picker-stub"),
    ).not.toBeInTheDocument();
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
    expect(
      screen.queryByTestId("agent-profile-picker-stub"),
    ).not.toBeInTheDocument();
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
      screen.queryByTestId("agent-profile-picker-stub"),
    ).not.toBeInTheDocument();
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
