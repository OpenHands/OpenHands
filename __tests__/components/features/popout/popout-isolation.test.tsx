import React from "react";
import { act, fireEvent, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { NavigationProvider } from "#/context/navigation-context";
import { ConversationRenderScopeProvider } from "#/contexts/conversation-render-scope";
import { InteractiveChatBox } from "#/components/features/chat/interactive-chat-box";
import { ConversationConfirmationButtons } from "#/components/shared/buttons/conversation-confirmation-buttons";
import { PopoutHost } from "#/components/features/popout/popout-host";
import { usePopoutStore } from "#/stores/popout-store";
import {
  getConversationExecutionStatus,
  useConversationStateStore,
} from "#/stores/conversation-state-store";
import {
  getComposerBucket,
  useConversationStore,
} from "#/stores/conversation-store";
import { useEventStore } from "#/stores/use-event-store";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { I18nKey } from "#/i18n/declaration";
import type { OpenHandsEvent } from "#/types/agent-server/core";
const PRIMARY_ID = "primary-conv";
const POPOUT_ID = "popout-conv";

vi.mock("#/contexts/websocket-provider-wrapper", () => ({
  WebSocketProviderWrapper: ({ children }: { children: React.ReactNode }) =>
    children,
}));

vi.mock("#/wrapper/event-handler", () => ({
  EventHandler: ({ children }: { children: React.ReactNode }) => children,
}));

// Keep PopoutConversation real; swap ChatInterface for the surfaces that
// previously leaked process-wide composer / execution state.
vi.mock("#/components/features/chat/chat-interface", async () => {
  const { TestChatSurface } = await import("./test-chat-surface");
  return { ChatInterface: TestChatSurface };
});

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: vi.fn(),
}));

vi.mock("#/hooks/query/use-user-conversation", () => ({
  useUserConversation: vi.fn(),
}));

vi.mock("#/hooks/chat/use-btw-interceptor", () => ({
  useBtwInterceptor: (_id: string | null, next: (message: string) => void) =>
    next,
}));

vi.mock("#/hooks/chat/use-goal-interceptor", () => ({
  useGoalInterceptor: (_id: string | null, next: (message: string) => void) =>
    next,
}));

vi.mock("#/hooks/chat/use-model-interceptor", () => ({
  useModelInterceptor: (_id: string | null, next: (message: string) => void) =>
    next,
}));

vi.mock("#/hooks/query/use-sub-conversation-task-polling", () => ({
  useSubConversationTaskPolling: () => ({ taskStatus: null }),
}));

vi.mock("#/hooks/mutation/use-respond-to-confirmation", () => ({
  useRespondToConfirmation: () => ({ mutate: vi.fn() }),
}));

vi.mock("#/hooks/chat/use-chat-attachment-upload", () => ({
  useChatAttachmentUpload: () => ({ handleUpload: vi.fn() }),
}));

vi.mock("#/components/features/chat/git-control-bar", () => ({
  GitControlBar: () => null,
}));

vi.mock("#/hooks/chat/use-slash-command", () => ({
  useSlashCommand: () => ({
    isMenuOpen: false,
    filteredItems: [],
    selectedIndex: 0,
    updateSlashMenu: vi.fn(),
    selectItem: vi.fn(),
    handleSlashKeyDown: () => false,
    closeMenu: vi.fn(),
  }),
}));

vi.mock("#/hooks/chat/use-draft-persistence", () => ({
  useDraftPersistence: () => ({
    saveDraft: vi.fn(),
    clearDraft: vi.fn(),
  }),
}));

import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUserConversation } from "#/hooks/query/use-user-conversation";

function PrimaryConversation() {
  return (
    <ConversationRenderScopeProvider isPrimary>
      <NavigationProvider
        value={{
          currentPath: `/conversations/${PRIMARY_ID}`,
          conversationId: PRIMARY_ID,
          isNavigating: false,
          navigate: vi.fn(),
        }}
      >
        <div data-testid="primary-conversation">
          <ConversationConfirmationButtons />
          <InteractiveChatBox onSubmit={vi.fn()} hasStartedConversation />
        </div>
      </NavigationProvider>
    </ConversationRenderScopeProvider>
  );
}

function mockConversation(id: string, executionStatus: ExecutionStatus | null) {
  return {
    data: {
      id,
      execution_status: executionStatus,
      conversation_url: `http://localhost/api/conversations/${id}`,
      session_api_key: null,
    },
    isFetched: true,
  } as ReturnType<typeof useActiveConversation>;
}

function seedAgentAction(conversationId: string) {
  const event = {
    id: `${conversationId}-action`,
    timestamp: new Date().toISOString(),
    source: "agent",
    thought: [],
    thinking_blocks: [],
    action: {
      kind: "ExecuteBashAction",
      command: "echo hi",
    },
    tool_name: "terminal",
    tool_call_id: `${conversationId}-call`,
    llm_response_id: null,
  } as unknown as OpenHandsEvent;

  useEventStore.getState().loadConversation(conversationId);
  useEventStore.getState().addEvent(conversationId, event);
}

describe("Popout conversation isolation", () => {
  beforeEach(() => {
    usePopoutStore.setState({ popouts: [] });
    useConversationStateStore.getState().reset();
    useConversationStore.setState({ byConversation: {} });
    useEventStore.getState().clearEvents();
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 1280,
    });

    vi.mocked(useActiveConversation).mockImplementation(() => {
      // Default: components resolve via navigation scope in real app; here the
      // mock cannot see context, so return a neutral idle conversation. Live
      // status from the keyed store is what the test asserts against.
      return mockConversation("unknown", ExecutionStatus.IDLE);
    });
    vi.mocked(useUserConversation).mockImplementation((id) =>
      mockConversation(id ?? "unknown", ExecutionStatus.IDLE),
    );
  });

  it("keeps execution status, confirmation UI, drafts, commands, and attachments isolated", () => {
    seedAgentAction(PRIMARY_ID);
    seedAgentAction(POPOUT_ID);

    usePopoutStore.getState().openPopout({
      conversationId: POPOUT_ID,
      title: "Branch",
    });

    renderWithProviders(
      <>
        <PrimaryConversation />
        <PopoutHost />
      </>,
      {
        navigation: {
          currentPath: `/conversations/${PRIMARY_ID}`,
          conversationId: PRIMARY_ID,
        },
      },
    );

    const primary = screen.getByTestId("primary-conversation");
    const popout = screen.getByTestId("popout-conversation");

    // Primary awaiting confirmation must not disable the popout composer or
    // show confirmation chrome against the popout's last agent event.
    act(() => {
      useConversationStateStore
        .getState()
        .setExecutionStatus(
          PRIMARY_ID,
          ExecutionStatus.WAITING_FOR_CONFIRMATION,
        );
      useConversationStateStore
        .getState()
        .setExecutionStatus(POPOUT_ID, ExecutionStatus.IDLE);
    });

    expect(
      getConversationExecutionStatus(
        useConversationStateStore.getState(),
        PRIMARY_ID,
      ),
    ).toBe(ExecutionStatus.WAITING_FOR_CONFIRMATION);
    expect(
      getConversationExecutionStatus(
        useConversationStateStore.getState(),
        POPOUT_ID,
      ),
    ).toBe(ExecutionStatus.IDLE);
    expect(
      within(primary).getByText(I18nKey.CHAT_INTERFACE$USER_ASK_CONFIRMATION),
    ).toBeInTheDocument();
    expect(
      within(popout).queryByText(I18nKey.CHAT_INTERFACE$USER_ASK_CONFIRMATION),
    ).not.toBeInTheDocument();

    // Flip statuses: only the popout should await confirmation.
    act(() => {
      useConversationStateStore
        .getState()
        .setExecutionStatus(PRIMARY_ID, ExecutionStatus.IDLE);
      useConversationStateStore
        .getState()
        .setExecutionStatus(
          POPOUT_ID,
          ExecutionStatus.WAITING_FOR_CONFIRMATION,
        );
    });

    expect(
      getConversationExecutionStatus(
        useConversationStateStore.getState(),
        PRIMARY_ID,
      ),
    ).toBe(ExecutionStatus.IDLE);
    expect(
      getConversationExecutionStatus(
        useConversationStateStore.getState(),
        POPOUT_ID,
      ),
    ).toBe(ExecutionStatus.WAITING_FOR_CONFIRMATION);
    expect(
      within(primary).queryByText(I18nKey.CHAT_INTERFACE$USER_ASK_CONFIRMATION),
    ).not.toBeInTheDocument();
    expect(
      within(popout).getByText(I18nKey.CHAT_INTERFACE$USER_ASK_CONFIRMATION),
    ).toBeInTheDocument();

    // Programmatic commands land in the matching composer only (applied then
    // one-shot consumed from the store). Attachments stay bucketed.
    const primaryFile = new File(["p"], "primary.txt", { type: "text/plain" });
    const popoutFile = new File(["o"], "popout.txt", { type: "text/plain" });

    act(() => {
      useConversationStore
        .getState()
        .setMessageToSend(PRIMARY_ID, "primary command");
      useConversationStore
        .getState()
        .setMessageToSend(POPOUT_ID, "popout command");
      useConversationStore.getState().addFiles(PRIMARY_ID, [primaryFile]);
      useConversationStore.getState().addFiles(POPOUT_ID, [popoutFile]);
    });

    expect(within(primary).getByTestId("chat-input")).toHaveTextContent(
      "primary command",
    );
    expect(within(popout).getByTestId("chat-input")).toHaveTextContent(
      "popout command",
    );
    expect(
      getComposerBucket(useConversationStore.getState(), PRIMARY_ID).files.map(
        (file) => file.name,
      ),
    ).toEqual(["primary.txt"]);
    expect(
      getComposerBucket(useConversationStore.getState(), POPOUT_ID).files.map(
        (file) => file.name,
      ),
    ).toEqual(["popout.txt"]);

    // Clearing / sending from the popout must not wipe the primary composer.
    act(() => {
      useConversationStore.getState().clearAllFiles(POPOUT_ID);
      useConversationStore.getState().setMessageToSend(POPOUT_ID, "");
    });

    expect(
      getComposerBucket(useConversationStore.getState(), PRIMARY_ID).files.map(
        (file) => file.name,
      ),
    ).toEqual(["primary.txt"]);
    expect(
      getComposerBucket(useConversationStore.getState(), POPOUT_ID).files,
    ).toEqual([]);
    expect(within(primary).getByTestId("chat-input")).toHaveTextContent(
      "primary command",
    );

    // Closing the popout drops its scoped buckets and leaves the primary alone.
    fireEvent.click(screen.getByRole("button", { name: I18nKey.POPOUT$CLOSE }));

    expect(usePopoutStore.getState().popouts).toEqual([]);
    expect(
      useConversationStateStore.getState().byConversation[POPOUT_ID],
    ).toBeUndefined();
    // Primary attachments / status survive popout teardown.
    expect(
      getComposerBucket(useConversationStore.getState(), PRIMARY_ID).files.map(
        (file) => file.name,
      ),
    ).toEqual(["primary.txt"]);
    expect(
      useConversationStateStore.getState().byConversation[PRIMARY_ID]
        ?.execution_status,
    ).toBe(ExecutionStatus.IDLE);
  });
});
