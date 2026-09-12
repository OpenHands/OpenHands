import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SharedConversation as SharedConversationData } from "@openhands/typescript-client";

import SharedConversation from "#/routes/shared-conversation";
import { I18nKey } from "#/i18n/declaration";
import {
  SecurityRisk,
  type ActionEvent,
  type MessageEvent,
  type ObservationEvent,
  type OpenHandsEvent,
  type SystemPromptEvent,
} from "#/types/agent-server/core";

const mocks = vi.hoisted(() => ({
  useSharedConversation: vi.fn(),
  useSharedConversationEvents: vi.fn(),
  useInfiniteScroll: vi.fn(),
  useTranslation: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: (namespace?: unknown) => {
    mocks.useTranslation(namespace);
    return {
      t: (key: string) => key,
      i18n: { language: "en", exists: () => false },
    };
  },
}));

vi.mock("#/hooks/query/use-shared-conversation", () => ({
  useSharedConversation: (conversationId?: string) =>
    mocks.useSharedConversation(conversationId),
}));

vi.mock("#/hooks/query/use-shared-conversation-events", () => ({
  useSharedConversationEvents: (conversationId?: string) =>
    mocks.useSharedConversationEvents(conversationId),
}));

vi.mock("#/hooks/use-infinite-scroll", () => ({
  useInfiniteScroll: (options: unknown) => mocks.useInfiniteScroll(options),
}));

vi.mock("#/components/conversation-events/chat/messages", () => ({
  Messages: ({
    messages,
    allEvents,
  }: {
    messages: OpenHandsEvent[];
    allEvents: OpenHandsEvent[];
  }) => (
    <section
      data-testid="shared-messages"
      data-renderable-event-ids={messages.map((event) => event.id).join(",")}
      data-all-event-ids={allEvents.map((event) => event.id).join(",")}
    />
  ),
}));

vi.mock("#/components/shared/loading-spinner", () => ({
  LoadingSpinner: ({ size }: { size: "small" | "large" }) => (
    <div data-testid={`loading-spinner-${size}`}>{size}</div>
  ),
}));

const SHARED_CONVERSATION_ID = "shared-conversation-1";

const createSharedConversation = (
  overrides: Partial<SharedConversationData> = {},
): SharedConversationData => ({
  id: SHARED_CONVERSATION_ID,
  created_by_user_id: "user-1",
  selected_repository: "OpenHands/agent-canvas",
  selected_branch: "feature/shared-viewer",
  git_provider: "github",
  title: "A shared debugging session",
  pr_number: [1688],
  llm_model: "openhands/claude-haiku-4-5-20251001",
  metrics: null,
  parent_conversation_id: null,
  sub_conversation_ids: [],
  created_at: "2026-07-13T12:00:00.000Z",
  updated_at: "2026-07-13T12:30:00.000Z",
  ...overrides,
});

const createMessageEvent = (
  id: string,
  text: string,
  source: "agent" | "user" = "user",
): MessageEvent => ({
  id,
  timestamp: "2026-07-13T12:00:00.000Z",
  source,
  llm_message: {
    role: source === "user" ? "user" : "assistant",
    content: [{ type: "text", text }],
  },
  activated_microagents: [],
  extended_content: [],
});

const createBashAction = (): ActionEvent => ({
  id: "bash-action",
  timestamp: "2026-07-13T12:01:00.000Z",
  source: "agent",
  thought: [{ type: "text", text: "Inspect the workspace" }],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: "pwd",
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "execute_bash",
  tool_call_id: "bash-call",
  tool_call: {
    id: "bash-call",
    type: "function",
    function: {
      name: "execute_bash",
      arguments: JSON.stringify({ command: "pwd" }),
    },
  },
  llm_response_id: "response-1",
  security_risk: SecurityRisk.UNKNOWN,
});

const createBashObservation = (): ObservationEvent => ({
  id: "bash-observation",
  timestamp: "2026-07-13T12:02:00.000Z",
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: "bash-call",
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: "/workspace/project\n" }],
    command: "pwd",
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {
      exit_code: 0,
      pid: 123,
      username: "openhands",
      hostname: "sandbox",
      working_dir: "/workspace/project",
      py_interpreter_path: null,
      prefix: "",
      suffix: "",
    },
  },
  action_id: "bash-action",
});

const createHiddenSystemPrompt = (): SystemPromptEvent => ({
  id: "system-prompt",
  timestamp: "2026-07-13T11:59:00.000Z",
  source: "agent",
  system_prompt: { type: "text", text: "Internal instructions" },
  tools: [],
});

interface SharedEventPage {
  items: OpenHandsEvent[];
  next_page_id: string | null;
}

interface SharedEventsData {
  pages: SharedEventPage[];
}

interface ConversationHookState {
  data: SharedConversationData | null | undefined;
  isLoading: boolean;
  error: Error | null;
}

interface EventsHookState {
  data: SharedEventsData | undefined;
  isLoading: boolean;
  error: Error | null;
  hasNextPage: boolean | undefined;
  fetchNextPage: () => void;
  isFetchingNextPage: boolean;
}

const createConversationHookState = (
  overrides: Partial<ConversationHookState> = {},
): ConversationHookState => ({
  data: createSharedConversation(),
  isLoading: false,
  error: null,
  ...overrides,
});

const createEventsHookState = (
  overrides: Partial<EventsHookState> = {},
): EventsHookState => ({
  data: { pages: [{ items: [], next_page_id: null }] },
  isLoading: false,
  error: null,
  hasNextPage: false,
  fetchNextPage: vi.fn(),
  isFetchingNextPage: false,
  ...overrides,
});

interface RenderViewerOptions {
  conversationState?: Partial<ConversationHookState>;
  eventsState?: Partial<EventsHookState>;
}

const renderViewer = ({
  conversationState: conversationOverrides = {},
  eventsState: eventsOverrides = {},
}: RenderViewerOptions = {}) => {
  const conversationState = createConversationHookState(conversationOverrides);
  const eventsState = createEventsHookState(eventsOverrides);
  const scrollContainerRef = createRef<HTMLDivElement>();

  mocks.useSharedConversation.mockReturnValue(conversationState);
  mocks.useSharedConversationEvents.mockReturnValue(eventsState);
  mocks.useInfiniteScroll.mockReturnValue(scrollContainerRef);

  const viewer = () => (
    <MemoryRouter
      initialEntries={[`/shared/conversations/${SHARED_CONVERSATION_ID}`]}
    >
      <Routes>
        <Route
          path="/shared/conversations/:conversationId"
          element={<SharedConversation />}
        />
      </Routes>
    </MemoryRouter>
  );
  const rendered = render(viewer());

  return {
    ...rendered,
    conversationState,
    eventsState,
    scrollContainerRef,
    rerenderViewer: () => rendered.rerender(viewer()),
  };
};

afterEach(() => {
  vi.clearAllMocks();
});

describe("shared conversation viewer", () => {
  it.each([
    {
      source: "conversation",
      conversationState: { isLoading: true },
      eventsState: {},
    },
    {
      source: "events",
      conversationState: {},
      eventsState: { isLoading: true },
    },
  ])(
    "shows the full-page loader while $source data is loading",
    ({ conversationState, eventsState }) => {
      renderViewer({ conversationState, eventsState });

      expect(screen.getByTestId("loading-spinner-large")).toBeInTheDocument();
      expect(screen.queryByRole("heading")).toBeNull();
    },
  );

  it.each([
    {
      reason: "conversation request fails",
      conversationState: { error: new Error("conversation unavailable") },
      eventsState: {},
    },
    {
      reason: "events request fails",
      conversationState: {},
      eventsState: { error: new Error("events unavailable") },
    },
    {
      reason: "conversation is missing",
      conversationState: { data: null },
      eventsState: {},
    },
  ])(
    "shows the not-found state when $reason",
    ({ conversationState, eventsState }) => {
      renderViewer({ conversationState, eventsState });

      expect(
        screen.getByText(I18nKey.CONVERSATION$NOT_FOUND),
      ).toBeInTheDocument();
      expect(screen.queryByTestId("shared-messages")).toBeNull();
    },
  );

  it("renders metadata and a reconstructed, filtered multipage event stream", () => {
    const userMessage = createMessageEvent(
      "user-message",
      "Please inspect the workspace",
    );
    const action = createBashAction();
    const observation = createBashObservation();
    const hiddenSystemPrompt = createHiddenSystemPrompt();
    const fetchNextPage = vi.fn();
    const { scrollContainerRef } = renderViewer({
      eventsState: {
        data: {
          pages: [
            {
              items: [hiddenSystemPrompt, userMessage, action],
              next_page_id: "page-2",
            },
            {
              items: [observation],
              next_page_id: null,
            },
          ],
        },
        hasNextPage: true,
        fetchNextPage,
        isFetchingNextPage: true,
      },
    });

    expect(
      screen.getByRole("heading", { name: "A shared debugging session" }),
    ).toBeInTheDocument();
    expect(mocks.useTranslation).toHaveBeenCalledWith("openhands");
    expect(
      screen.getByText(`${I18nKey.CONVERSATION$BRANCH}: feature/shared-viewer`),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        `${I18nKey.CONVERSATION$REPOSITORY}: OpenHands/agent-canvas`,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        `${I18nKey.LLM$MODEL}: openhands/claude-haiku-4-5-20251001`,
      ),
    ).toBeInTheDocument();

    const homeLink = screen.getByRole("link", {
      name: I18nKey.BRANDING$OPENHANDS_LOGO,
    });
    expect(homeLink).toHaveAttribute("href", "/conversations");

    const messages = screen.getByTestId("shared-messages");
    expect(messages).toHaveAttribute(
      "data-renderable-event-ids",
      "user-message,bash-observation",
    );
    expect(messages).toHaveAttribute(
      "data-all-event-ids",
      "system-prompt,user-message,bash-action,bash-observation",
    );
    expect(screen.getByTestId("loading-spinner-small")).toBeInTheDocument();

    expect(mocks.useSharedConversation).toHaveBeenCalledWith(
      SHARED_CONVERSATION_ID,
    );
    expect(mocks.useSharedConversationEvents).toHaveBeenCalledWith(
      SHARED_CONVERSATION_ID,
    );
    expect(mocks.useInfiniteScroll).toHaveBeenCalledWith({
      hasNextPage: true,
      isFetchingNextPage: true,
      fetchNextPage,
    });
    expect(scrollContainerRef.current).toHaveClass("overflow-y-auto");
  });

  it("replaces the visible event stream when fetched pages change", () => {
    const initialMessage = createMessageEvent("initial-message", "First page");
    const replacementMessage = createMessageEvent(
      "replacement-message",
      "Replacement page",
    );
    const { rerenderViewer } = renderViewer({
      eventsState: {
        data: {
          pages: [{ items: [initialMessage], next_page_id: "page-2" }],
        },
      },
    });

    expect(screen.getByTestId("shared-messages")).toHaveAttribute(
      "data-renderable-event-ids",
      "initial-message",
    );

    mocks.useSharedConversationEvents.mockReturnValue(
      createEventsHookState({
        data: {
          pages: [{ items: [replacementMessage], next_page_id: null }],
        },
      }),
    );
    rerenderViewer();

    expect(screen.getByTestId("shared-messages")).toHaveAttribute(
      "data-renderable-event-ids",
      "replacement-message",
    );
    expect(screen.getByTestId("shared-messages")).toHaveAttribute(
      "data-all-event-ids",
      "replacement-message",
    );
  });

  it("uses fallback copy and hides optional details when history is unavailable", () => {
    renderViewer({
      conversationState: {
        data: createSharedConversation({
          title: null,
          selected_branch: null,
          selected_repository: null,
          llm_model: null,
        }),
      },
      eventsState: {
        data: undefined,
        hasNextPage: undefined,
      },
    });

    expect(
      screen.getByRole("heading", {
        name: I18nKey.CONVERSATION$SHARED_CONVERSATION,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.CONVERSATION$NO_HISTORY_AVAILABLE),
    ).toBeInTheDocument();
    expect(screen.queryByText(I18nKey.CONVERSATION$BRANCH)).toBeNull();
    expect(screen.queryByText(I18nKey.CONVERSATION$REPOSITORY)).toBeNull();
    expect(screen.queryByText(I18nKey.LLM$MODEL)).toBeNull();
    expect(screen.queryByTestId("shared-messages")).toBeNull();
    expect(screen.queryByTestId("loading-spinner-small")).toBeNull();
    expect(mocks.useInfiniteScroll).toHaveBeenCalledWith(
      expect.objectContaining({ hasNextPage: false }),
    );
  });
});
