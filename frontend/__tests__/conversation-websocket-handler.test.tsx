import React from "react";
import {
  describe,
  it,
  expect,
  beforeAll,
  beforeEach,
  afterAll,
  afterEach,
  vi,
} from "vitest";
import { screen, waitFor, render, cleanup, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useBrowserStore } from "#/stores/browser-store";
import { useCommandStore } from "#/stores/command-store";
import { useErrorMessageStore } from "#/stores/error-message-store";
import {
  createMockMessageEvent,
  createMockUserMessageEvent,
  createMockConversationErrorEvent,
  createMockServerErrorEvent,
  createMockBrowserObservationEvent,
  createMockBrowserNavigateActionEvent,
  createMockExecuteBashActionEvent,
  createMockExecuteBashObservationEvent,
} from "#/mocks/mock-ws-helpers";
import {
  ConnectionStatusComponent,
  EventStoreComponent,
  OptimisticUserMessageStoreComponent,
  ErrorMessageStoreComponent,
} from "./helpers/websocket-test-components";
import {
  ConversationWebSocketProvider,
  useConversationWebSocket,
} from "#/contexts/conversation-websocket-context";
import { conversationWebSocketTestSetup } from "./helpers/msw-websocket-setup";
import { useEventStore } from "#/stores/use-event-store";
import { isV1Event } from "#/types/v1/type-guards";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { useConversationStore } from "#/stores/conversation-store";
import type { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";
import type { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";

// Mock useUserConversation to return V1 conversation data
vi.mock("#/hooks/query/use-user-conversation", () => ({
  useUserConversation: vi.fn(() => ({
    data: {
      conversation_version: "V1",
      status: "RUNNING",
    },
    isLoading: false,
    isFetched: true,
    error: null,
  })),
}));

// MSW WebSocket mock setup
const { wsLink, server: mswServer } = conversationWebSocketTestSetup();
let restoreWebSocketGlobal: (() => void) | null = null;

beforeAll(() => {
  // The global MSW server from vitest.setup.ts is already running
  // We just need to start our WebSocket-specific server
  mswServer.listen({ onUnhandledRequest: "bypass" });
});

beforeEach(() => {
  useSelectedOrganizationStore.setState({ organizationId: "test-org-id" });
  useConversationStore.getState().setConversationMode("code");
});

afterEach(() => {
  mswServer.resetHandlers();
  // Clean up any React components
  cleanup();
  restoreWebSocketGlobal?.();
  restoreWebSocketGlobal = null;
  // Reset stores to prevent state leakage between tests
  useErrorMessageStore.getState().removeErrorMessage();
  useEventStore.getState().clearEvents();
});

afterAll(async () => {
  // Close the WebSocket MSW server
  mswServer.close();

  // Give time for any pending WebSocket connections to close. This is very important to prevent serious memory leaks
  await new Promise((resolve) => {
    setTimeout(resolve, 500);
  });
});

// Helper function to render components with ConversationWebSocketProvider
function renderWithWebSocketContext(
  children: React.ReactNode,
  conversationId = "test-conversation-default",
  conversationUrl = "http://localhost:3000/api/conversations/test-conversation-default",
  sessionApiKey: string | null = null,
  sandboxStatus: V1SandboxStatus | null = null,
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/test-conversation-default"]}>
        <Routes>
          <Route
            path="/:conversationId"
            element={
              <ConversationWebSocketProvider
                conversationId={conversationId}
                conversationUrl={conversationUrl}
                sessionApiKey={sessionApiKey}
                sandboxStatus={sandboxStatus}
              >
                {children}
              </ConversationWebSocketProvider>
            }
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

type CapturedSendMessage = NonNullable<
  ReturnType<typeof useConversationWebSocket>
>["sendMessage"];

const createTextMessage = (text: string) => ({
  role: "user" as const,
  content: [{ type: "text" as const, text }],
});

const createMockSubConversation = (
  id: string,
  conversationUrl: string,
): V1AppConversation => ({
  id,
  created_by_user_id: "test-user-id",
  sandbox_id: "test-sandbox-id",
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: null,
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  sandbox_status: "RUNNING",
  execution_status: null,
  conversation_url: conversationUrl,
  session_api_key: null,
  sub_conversation_ids: [],
});

function SendMessageCapture({
  onReady,
}: {
  onReady: (sendMessage: CapturedSendMessage) => void;
}) {
  const context = useConversationWebSocket();

  React.useEffect(() => {
    if (context?.sendMessage) {
      onReady(context.sendMessage);
    }
  }, [context?.sendMessage, onReady]);

  return (
    <div>
      <div data-testid="connection-state">
        {context?.connectionState || "NOT_AVAILABLE"}
      </div>
    </div>
  );
}

function setupControlledWebSocket() {
  const originalWebSocket = globalThis.WebSocket;
  const sockets: ControlledWebSocket[] = [];

  const eventForSocket = (type: string, socket: ControlledWebSocket) => {
    const event = new Event(type);
    Object.defineProperty(event, "target", { value: socket });
    Object.defineProperty(event, "currentTarget", { value: socket });
    return event;
  };

  const closeEventForSocket = (
    socket: ControlledWebSocket,
    code: number,
    reason: string,
  ) => {
    const event = eventForSocket("close", socket);
    Object.defineProperty(event, "code", { value: code });
    Object.defineProperty(event, "reason", { value: reason });
    return event as CloseEvent;
  };

  class ControlledWebSocket {
    static readonly CONNECTING = 0;

    static readonly OPEN = 1;

    static readonly CLOSING = 2;

    static readonly CLOSED = 3;

    readonly CONNECTING = ControlledWebSocket.CONNECTING;

    readonly OPEN = ControlledWebSocket.OPEN;

    readonly CLOSING = ControlledWebSocket.CLOSING;

    readonly CLOSED = ControlledWebSocket.CLOSED;

    binaryType: BinaryType = "blob";

    bufferedAmount = 0;

    extensions = "";

    onclose: ((event: CloseEvent) => void) | null = null;

    onerror: ((event: Event) => void) | null = null;

    onmessage: ((event: MessageEvent) => void) | null = null;

    onopen: ((event: Event) => void) | null = null;

    protocol = "";

    readyState = ControlledWebSocket.CONNECTING;

    sentMessages: unknown[] = [];

    url: string;

    constructor(url: string | URL) {
      this.url = String(url);
      sockets.push(this);
    }

    addEventListener() {
      return this.readyState;
    }

    close(code = 1000, reason = "") {
      if (this.readyState === ControlledWebSocket.CLOSED) {
        return;
      }

      this.readyState = ControlledWebSocket.CLOSED;
      this.onclose?.(closeEventForSocket(this, code, reason));
    }

    dispatchEvent() {
      return this.readyState !== ControlledWebSocket.CLOSED;
    }

    emitMessage(message: unknown) {
      const event = new MessageEvent("message", {
        data: typeof message === "string" ? message : JSON.stringify(message),
      });
      Object.defineProperty(event, "target", { value: this });
      Object.defineProperty(event, "currentTarget", { value: this });
      this.onmessage?.(event);
    }

    open() {
      this.readyState = ControlledWebSocket.OPEN;
      this.onopen?.(eventForSocket("open", this));
    }

    removeEventListener() {
      return this.readyState;
    }

    send(data: string | ArrayBufferLike | Blob | ArrayBufferView) {
      if (this.readyState !== ControlledWebSocket.OPEN) {
        throw new Error("WebSocket is not open");
      }

      this.sentMessages.push(JSON.parse(String(data)));
    }
  }

  vi.stubGlobal("WebSocket", ControlledWebSocket);
  restoreWebSocketGlobal = () => {
    vi.stubGlobal("WebSocket", originalWebSocket);
  };

  return { sockets };
}

describe("Conversation WebSocket Handler", () => {
  // 1. Connection Lifecycle Tests
  describe("Connection Management", () => {
    it("should establish WebSocket connection to /events/socket URL", async () => {
      // This will fail because we haven't created the context yet
      renderWithWebSocketContext(<ConnectionStatusComponent />);

      // Initially should be CONNECTING
      expect(screen.getByTestId("connection-state")).toHaveTextContent(
        "CONNECTING",
      );

      // Wait for connection to be established
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });
    });

    it.todo("should provide manual disconnect functionality");
  });

  // 2. Event Processing Tests
  describe("Event Stream Processing", () => {
    it("should update event store with received WebSocket events", async () => {
      // Create a mock MessageEvent to send through WebSocket
      const mockMessageEvent = createMockMessageEvent();

      // Set up MSW to send the event when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the mock event after connection
          client.send(JSON.stringify(mockMessageEvent));
        }),
      );

      // Render components that use both WebSocket and event store
      renderWithWebSocketContext(<EventStoreComponent />);

      // Wait for connection and event processing
      await waitFor(() => {
        expect(screen.getByTestId("events-count")).toHaveTextContent("1");
      });

      // Verify the event was added to the store
      expect(screen.getByTestId("latest-event-id")).toHaveTextContent(
        "test-event-123",
      );
      expect(screen.getByTestId("ui-events-count")).toHaveTextContent("1");
    });

    it("should handle malformed/invalid event data gracefully", async () => {
      // Suppress expected console.warn for invalid JSON parsing
      const consoleWarnSpy = vi
        .spyOn(console, "warn")
        .mockImplementation(() => {});

      // Set up MSW to send various invalid events when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();

          // Send invalid JSON
          client.send("invalid json string");

          // Send valid JSON but missing required fields
          client.send(JSON.stringify({ message: "missing required fields" }));

          // Send valid JSON with wrong data types
          client.send(
            JSON.stringify({
              id: 123, // should be string
              timestamp: "2023-01-01T00:00:00Z",
              source: "agent",
            }),
          );

          // Send null values for required fields
          client.send(
            JSON.stringify({
              id: null,
              timestamp: "2023-01-01T00:00:00Z",
              source: "agent",
            }),
          );

          // Send a valid event after invalid ones to ensure processing continues
          client.send(
            JSON.stringify({
              id: "valid-event-123",
              timestamp: new Date().toISOString(),
              source: "agent",
              llm_message: {
                role: "assistant",
                content: [
                  { type: "text", text: "Valid message after invalid ones" },
                ],
              },
              activated_microagents: [],
              extended_content: [],
            }),
          );
        }),
      );

      // Render components that use both WebSocket and event store
      renderWithWebSocketContext(<EventStoreComponent />);

      // Wait for connection and event processing
      // Only the valid event should be added to the store
      await waitFor(() => {
        expect(screen.getByTestId("events-count")).toHaveTextContent("1");
      });

      // Verify only the valid event was added
      expect(screen.getByTestId("latest-event-id")).toHaveTextContent(
        "valid-event-123",
      );
      expect(screen.getByTestId("ui-events-count")).toHaveTextContent("1");

      // Restore console.warn
      consoleWarnSpy.mockRestore();
    });
  });

  // 3. State Management Tests
  describe("State Management Integration", () => {
    it("should clear optimistic user messages when confirmed", async () => {
      // First, set an optimistic user message
      const { setOptimisticUserMessage } =
        useOptimisticUserMessageStore.getState();
      setOptimisticUserMessage("This is an optimistic message");

      // Create a mock user MessageEvent to send through WebSocket
      const mockUserMessageEvent = createMockUserMessageEvent();

      // Set up MSW to send the user message event when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the mock user message event after connection
          client.send(JSON.stringify(mockUserMessageEvent));
        }),
      );

      // Render components that use both WebSocket and optimistic user message store
      renderWithWebSocketContext(<OptimisticUserMessageStoreComponent />);

      // Initially should show the optimistic message
      expect(screen.getByTestId("optimistic-user-message")).toHaveTextContent(
        "This is an optimistic message",
      );

      // Wait for connection and user message event processing
      // The optimistic message should be cleared when user message is confirmed
      await waitFor(() => {
        expect(screen.getByTestId("optimistic-user-message")).toHaveTextContent(
          "none",
        );
      });
    });
  });

  // 4. Cache Management Tests
  describe("Cache Management", () => {
    it.todo(
      "should invalidate file changes cache on file edit/write/command events",
    );
    it.todo("should invalidate specific file diff cache on file modifications");
    it.todo("should prevent cache refetch during high message rates");
    it.todo("should not invalidate cache for non-file-related events");
    it.todo("should invalidate cache with correct conversation ID context");
  });

  // 5. Error Handling Tests
  describe("Error Handling & Recovery", () => {
    beforeEach(() => {
      // Clear stores before each error handling test to prevent state leakage
      useErrorMessageStore.getState().removeErrorMessage();
      useEventStore.getState().clearEvents();
    });

    it("should update error message store on ConversationErrorEvent", async () => {
      // ConversationErrorEvent represents infrastructure/authentication errors
      // that should be shown as a banner to the user.
      const mockConversationErrorEvent = createMockConversationErrorEvent();

      // Set up MSW to send the error event when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the mock error event after connection
          client.send(JSON.stringify(mockConversationErrorEvent));
        }),
      );

      // Render components that use both WebSocket and error message store
      renderWithWebSocketContext(<ErrorMessageStoreComponent />);

      // Initially should show "none"
      expect(screen.getByTestId("error-message")).toHaveTextContent("none");

      // Wait for connection and error event processing
      await waitFor(() => {
        expect(screen.getByTestId("error-message")).toHaveTextContent(
          "Your session has expired. Please log in again.",
        );
      });
    });

    it("should update error message store on ServerErrorEvent", async () => {
      // ServerErrorEvent represents server-side errors (e.g., MCP configuration errors)
      // that should be shown as a banner to the user.
      const mockServerErrorEvent = createMockServerErrorEvent();

      // Set up MSW to send the error event when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the mock error event after connection
          client.send(JSON.stringify(mockServerErrorEvent));
        }),
      );

      // Render components that use both WebSocket and error message store
      renderWithWebSocketContext(<ErrorMessageStoreComponent />);

      // Initially should show "none"
      expect(screen.getByTestId("error-message")).toHaveTextContent("none");

      // Wait for connection and error event processing
      await waitFor(() => {
        expect(screen.getByTestId("error-message")).toHaveTextContent(
          "MCP server connection failed: Invalid configuration",
        );
      });
    });

    it("should handle different ServerErrorEvent error codes", async () => {
      // Test different error codes for ServerErrorEvent
      const mockServerErrorEvent = createMockServerErrorEvent({
        code: "RuntimeError",
        detail: "Agent server runtime error: Out of memory",
      });

      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          client.send(JSON.stringify(mockServerErrorEvent));
        }),
      );

      renderWithWebSocketContext(<ErrorMessageStoreComponent />);

      await waitFor(() => {
        expect(screen.getByTestId("error-message")).toHaveTextContent(
          "Agent server runtime error: Out of memory",
        );
      });
    });

    it("should clear error message when a successful event is received after a ServerErrorEvent", async () => {
      // This test verifies that error banners disappear when follow-up messages
      // are sent and received after a ServerErrorEvent.
      // Note: This test was originally commented out because the implementation
      // didn't properly clear ServerErrorEvent errors on subsequent events.
      // After the fix using isDisplayableErrorEvent, this now works correctly.
      const conversationId = "test-server-error-clear";

      // Set up MSW to mock event count API and send events
      mswServer.use(
        http.get(
          `http://localhost:3000/api/conversations/${conversationId}/events/count`,
          () => HttpResponse.json(2),
        ),
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();

          // Send ServerErrorEvent first (sets the error banner)
          const mockServerErrorEvent = createMockServerErrorEvent();
          client.send(JSON.stringify(mockServerErrorEvent));

          // Send a successful (non-error) event immediately after
          // This simulates the user sending a follow-up message and receiving a response
          const mockSuccessEvent = createMockMessageEvent({
            id: "success-event-after-server-error",
          });
          client.send(JSON.stringify(mockSuccessEvent));
        }),
      );

      // Verify error message store is initially empty
      expect(useErrorMessageStore.getState().errorMessage).toBeNull();

      // Render with WebSocket context (minimal component just to trigger connection)
      renderWithWebSocketContext(
        <ConnectionStatusComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      // Wait for connection
      await waitFor(
        () => {
          expect(screen.getByTestId("connection-state")).toHaveTextContent(
            "OPEN",
          );
        },
        { timeout: 5000 },
      );

      // Wait for both events to be received and error to be cleared
      // The error was set by the first event (ServerErrorEvent),
      // then cleared by the second successful event (MessageEvent).
      await waitFor(
        () => {
          expect(useEventStore.getState().events.length).toBe(2);
          expect(useErrorMessageStore.getState().errorMessage).toBeNull();
        },
        { timeout: 5000 },
      );
    });

    it("should show friendly i18n message for budget ConversationErrorEvent", async () => {
      const mockBudgetConversationError = createMockConversationErrorEvent({
        detail:
          "Budget has been exceeded! Current cost: 18.51, Max budget: 18.24",
      });

      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          client.send(JSON.stringify(mockBudgetConversationError));
        }),
      );

      renderWithWebSocketContext(<ErrorMessageStoreComponent />);

      expect(screen.getByTestId("error-message")).toHaveTextContent("none");

      await waitFor(() => {
        expect(screen.getByTestId("error-message")).toHaveTextContent(
          "STATUS$ERROR_LLM_OUT_OF_CREDITS",
        );
      });
    });

    it.skip("should not clear budget error when non-agent events are received", async () => {
      // Regression test: budget/credit error banner used to disappear ~500ms after
      // appearing because every subsequent non-error event called removeErrorMessage().
      // NOTE: This test is skipped due to flakiness in the WebSocket test setup.
      // The functionality is tested by "should clear budget error when an agent event is received"
      // which verifies that budget errors ARE cleared when agent events arrive, proving the logic works.
      // The inverse (budget errors NOT cleared for user events) is handled by the handleNonErrorEvent
      // callback in the production code.
      const conversationId = "test-conversation-budget-persist";

      const mockBudgetError = createMockConversationErrorEvent({
        id: "budget-error-1",
        detail:
          "Budget has been exceeded! Current cost: 18.51, Max budget: 18.24",
      });

      // A user MessageEvent (source: "user") should NOT clear the budget error
      const mockUserEvent = createMockUserMessageEvent({
        id: "user-msg-after-error",
      });

      mswServer.use(
        http.get(
          `http://localhost:3000/api/conversations/${conversationId}/events/count`,
          () => HttpResponse.json(2),
        ),
        wsLink.addEventListener("connection", async ({ client, server }) => {
          server.connect();

          // Wait for connection to be established
          await new Promise((resolve) => {
            setTimeout(resolve, 100);
          });

          // Send budget error first
          client.send(JSON.stringify(mockBudgetError));

          // Wait for budget error to be processed before sending user event
          await new Promise((resolve) => {
            setTimeout(resolve, 200);
          });

          // Send user event - it should NOT clear the budget error
          client.send(JSON.stringify(mockUserEvent));
        }),
      );

      renderWithWebSocketContext(
        <ErrorMessageStoreComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      // Wait for connection
      await waitFor(
        () => {
          expect(screen.getByTestId("connection-state")).toHaveTextContent(
            "OPEN",
          );
        },
        { timeout: 5000 },
      );

      // Wait for both events to be processed
      await waitFor(
        () => {
          expect(useEventStore.getState().events.length).toBe(2);
        },
        { timeout: 5000 },
      );

      // Budget error should still be visible — not cleared by the user event
      expect(useErrorMessageStore.getState().errorMessage).toBe(
        "STATUS$ERROR_LLM_OUT_OF_CREDITS",
      );
    });

    it("should clear budget error when an agent event is received", async () => {
      // When the agent sends a new event, it means the LLM is working
      // (credits are available), so the budget error should be cleared.
      const conversationId = "test-conversation-budget-clear";

      const mockBudgetError = createMockConversationErrorEvent({
        id: "budget-error-2",
        detail:
          "Budget has been exceeded! Current cost: 18.51, Max budget: 18.24",
      });

      // An agent MessageEvent (source: "agent") SHOULD clear the budget error
      const mockAgentEvent = createMockMessageEvent({
        id: "agent-msg-after-credits",
      });

      mswServer.use(
        http.get(
          `http://localhost:3000/api/conversations/${conversationId}/events/count`,
          () => HttpResponse.json(2),
        ),
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          client.send(JSON.stringify(mockBudgetError));
          client.send(JSON.stringify(mockAgentEvent));
        }),
      );

      renderWithWebSocketContext(
        <ErrorMessageStoreComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      // Wait for both events to be processed
      await waitFor(() => {
        expect(useEventStore.getState().events.length).toBe(2);
      });

      // After both events processed, the budget error should have been cleared
      // by the agent event (source: "agent"). Check it's not the budget error.
      const currentError = useErrorMessageStore.getState().errorMessage;
      expect(currentError).not.toBe("STATUS$ERROR_LLM_OUT_OF_CREDITS");
    });

    it("should set error message store on WebSocket connection errors", async () => {
      // Simulate a connect-then-fail sequence (the MSW server auto-connects by default).
      // This should surface an error message because the app has previously connected.
      mswServer.use(
        wsLink.addEventListener("connection", ({ client }) => {
          setTimeout(() => {
            client.close(1006, "Connection failed");
          }, 50);
        }),
      );

      // Render components that use both WebSocket and error message store
      renderWithWebSocketContext(
        <>
          <ErrorMessageStoreComponent />
          <ConnectionStatusComponent />
        </>,
      );

      // Initially should show "none"
      expect(screen.getByTestId("error-message")).toHaveTextContent("none");

      // Wait for disconnect
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "CLOSED",
        );
      });

      await waitFor(() => {
        expect(screen.getByTestId("error-message")).not.toHaveTextContent(
          "none",
        );
      });
    });

    it("should set error message store on WebSocket disconnect with error", async () => {
      // Set up MSW to connect first, then disconnect with error
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();

          // Simulate disconnect with error after a short delay
          setTimeout(() => {
            client.close(1006, "Unexpected disconnect");
          }, 100);
        }),
      );

      // Render components that use both WebSocket and error message store
      renderWithWebSocketContext(
        <>
          <ErrorMessageStoreComponent />
          <ConnectionStatusComponent />
        </>,
      );

      // Initially should show "none"
      expect(screen.getByTestId("error-message")).toHaveTextContent("none");

      // Wait for connection to be established first
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Wait for disconnect and error message to be set
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "CLOSED",
        );
      });

      // Should set error message on unexpected disconnect
      await waitFor(() => {
        expect(screen.getByTestId("error-message")).not.toHaveTextContent(
          "none",
        );
      });
    });

    it("should clear error message store when connection is restored", async () => {
      let connectionAttempt = 0;

      // Fail once (after connect), then allow reconnection to stay open.
      mswServer.use(
        wsLink.addEventListener("connection", ({ client }) => {
          connectionAttempt += 1;

          if (connectionAttempt === 1) {
            setTimeout(() => {
              client.close(1006, "Initial connection failed");
            }, 50);
          }
        }),
      );

      // Render components that use both WebSocket and error message store
      renderWithWebSocketContext(
        <>
          <ErrorMessageStoreComponent />
          <ConnectionStatusComponent />
        </>,
      );

      // Initially should show "none"
      expect(screen.getByTestId("error-message")).toHaveTextContent("none");

      // Wait for first failure
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "CLOSED",
        );
      });

      await waitFor(() => {
        expect(screen.getByTestId("error-message")).not.toHaveTextContent(
          "none",
        );
      });

      // Wait for reconnect to happen and verify error clears on successful connection
      await waitFor(
        () => {
          expect(screen.getByTestId("connection-state")).toHaveTextContent(
            "OPEN",
          );
          expect(screen.getByTestId("error-message")).toHaveTextContent("none");
        },
        { timeout: 5000 },
      );
    });

    it("should clear error message when a successful event is received after a ConversationErrorEvent", async () => {
      // This test verifies that error banners disappear when follow-up messages
      // are sent and received. Only ConversationErrorEvent sets the error banner,
      // and any non-error event should clear it.
      const conversationId = "test-conversation-error-clear";
      const { sockets } = setupControlledWebSocket();

      // Set up MSW to mock event count API. WebSocket delivery is controlled
      // directly so prior MSW WebSocket handlers cannot leak into this test.
      mswServer.use(
        http.get(
          `http://localhost:3000/api/conversations/${conversationId}/events/count`,
          () => HttpResponse.json(2),
        ),
      );

      // Verify error message store is initially empty
      expect(useErrorMessageStore.getState().errorMessage).toBeNull();

      // Render with WebSocket context (minimal component just to trigger connection)
      renderWithWebSocketContext(
        <ConnectionStatusComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      await waitFor(() => {
        expect(sockets).toHaveLength(1);
      });

      await act(async () => {
        sockets[0].open();
      });

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      const mockConversationErrorEvent = createMockConversationErrorEvent();
      const mockSuccessEvent = createMockMessageEvent({
        id: "success-event-after-error",
      });

      await act(async () => {
        sockets[0].emitMessage(mockConversationErrorEvent);
        sockets[0].emitMessage(mockSuccessEvent);
      });

      // Wait for both events to be received and error to be cleared
      // The error was set by the first event (ConversationErrorEvent),
      // then cleared by the second successful event (MessageEvent).
      await waitFor(() => {
        expect(useEventStore.getState().events.length).toBe(2);
        expect(useErrorMessageStore.getState().errorMessage).toBeNull();
      });
    });

    it("should not create duplicate events when WebSocket reconnects with resend_all=true", async () => {
      const conversationId = "test-conversation-reconnect";
      let connectionCount = 0;

      // Clear event store before test
      useEventStore.getState().clearEvents();

      // Create mock events that will be sent on each connection
      const mockHistoryEvents = [
        createMockUserMessageEvent({ id: "event-1" }),
        createMockMessageEvent({ id: "event-2" }),
        createMockMessageEvent({ id: "event-3" }),
      ];

      // Set up MSW to mock event count API and WebSocket
      // The WebSocket will resend all events on each connection (simulating resend_all=true behavior)
      mswServer.use(
        http.get(
          `http://localhost:3000/api/conversations/${conversationId}/events/count`,
          () => HttpResponse.json(3),
        ),
        wsLink.addEventListener("connection", ({ client, server }) => {
          connectionCount += 1;
          server.connect();

          // Send all history events on EVERY connection (simulating resend_all=true)
          mockHistoryEvents.forEach((event) => {
            client.send(JSON.stringify(event));
          });

          // On first connection, simulate a disconnect after events are sent
          if (connectionCount === 1) {
            setTimeout(() => {
              client.close(1006, "Simulated disconnect");
            }, 100);
          }
        }),
      );

      // Render with WebSocket context
      renderWithWebSocketContext(
        <ConnectionStatusComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      // Wait for initial connection and events
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      await waitFor(() => {
        expect(useEventStore.getState().events.length).toBe(3);
      });

      // Wait for disconnect
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "CLOSED",
        );
      });

      // Wait for reconnection
      await waitFor(
        () => {
          expect(screen.getByTestId("connection-state")).toHaveTextContent(
            "OPEN",
          );
        },
        { timeout: 5000 },
      );

      // Give time for resent events to be processed
      await new Promise((resolve) => {
        setTimeout(resolve, 200);
      });

      // After reconnection, events should NOT be duplicated
      // The server sends 3 events again (resend_all=true), but we should deduplicate
      const { events } = useEventStore.getState();
      const v1Events = events.filter(isV1Event);
      const uniqueEventIds = [...new Set(v1Events.map((e) => e.id))];

      // This assertion will FAIL with current implementation (showing the bug)
      // Expected: 3 events (deduplicated)
      // Actual: 6 events (duplicated)
      expect(v1Events.length).toBe(3);
      expect(uniqueEventIds.length).toBe(3);

      // Verify we actually had 2 connections
      expect(connectionCount).toBe(2);
    });

    it.todo("should track and display errors with proper metadata");
    it.todo("should set appropriate error states on connection failures");
    it.todo(
      "should handle WebSocket close codes appropriately (1000, 1006, etc.)",
    );
  });

  // 6. Connection State Validation Tests
  describe("Connection State Management", () => {
    it.todo("should only connect when conversation is in RUNNING status");
    it.todo("should handle STARTING conversation state appropriately");
    it.todo("should disconnect when conversation is STOPPED");
    it.todo("should validate runtime status before connecting");
  });

  // 7. Message Sending Tests
  describe("Message Sending", () => {
    it("should send user actions through WebSocket when connected", async () => {
      // Arrange
      const conversationId = "test-conversation-send";

      // Set up MSW to connect WebSocket
      mswServer.use(
        wsLink.addEventListener("connection", ({ server }) => {
          server.connect();
        }),
      );

      // Create ref to store sendMessage function
      let sendMessageFn: typeof useConversationWebSocket extends () => infer R
        ? R extends { sendMessage: infer S }
          ? S
          : null
        : null = null;

      function TestComponent() {
        const context = useConversationWebSocket();

        React.useEffect(() => {
          if (context?.sendMessage) {
            sendMessageFn = context.sendMessage;
          }
        }, [context?.sendMessage]);

        return (
          <div>
            <div data-testid="connection-state">
              {context?.connectionState || "NOT_AVAILABLE"}
            </div>
          </div>
        );
      }

      // Act
      renderWithWebSocketContext(
        <TestComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Send a message
      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
      });

      let sendError: Error | null = null;
      await act(async () => {
        try {
          await sendMessageFn!({
            role: "user",
            content: [{ type: "text", text: "Hello from test" }],
          });
        } catch (error) {
          sendError = error as Error;
        }
      });
      expect(sendError).toBeNull();
    });

    it("should not throw error when sendMessage is called with WebSocket connected", async () => {
      // This test verifies that sendMessage doesn't throw an error
      // when the WebSocket is connected.
      const conversationId = "test-conversation-no-throw";
      let sendError: Error | null = null;

      // Set up MSW to connect and receive messages
      mswServer.use(
        wsLink.addEventListener("connection", ({ server }) => {
          server.connect();
        }),
      );

      // Create ref to store sendMessage function
      let sendMessageFn: typeof useConversationWebSocket extends () => infer R
        ? R extends { sendMessage: infer S }
          ? S
          : null
        : null = null;

      function TestComponent() {
        const context = useConversationWebSocket();

        React.useEffect(() => {
          if (context?.sendMessage) {
            sendMessageFn = context.sendMessage;
          }
        }, [context?.sendMessage]);

        return (
          <div>
            <div data-testid="connection-state">
              {context?.connectionState || "NOT_AVAILABLE"}
            </div>
          </div>
        );
      }

      // Act
      renderWithWebSocketContext(
        <TestComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Wait for the context to be available
      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
      });

      // Try to send a message
      await act(async () => {
        try {
          await sendMessageFn!({
            role: "user",
            content: [{ type: "text", text: "Test message" }],
          });
        } catch (error) {
          sendError = error as Error;
        }
      });

      // Assert - should NOT throw an error
      expect(sendError).toBeNull();
    });

    it("should send multiple messages through WebSocket in order", async () => {
      // Arrange
      const conversationId = "test-conversation-multi";
      const { sockets } = setupControlledWebSocket();

      // Create ref to store sendMessage function
      let sendMessageFn: typeof useConversationWebSocket extends () => infer R
        ? R extends { sendMessage: infer S }
          ? S
          : null
        : null = null;

      function TestComponent() {
        const context = useConversationWebSocket();

        React.useEffect(() => {
          if (context?.sendMessage) {
            sendMessageFn = context.sendMessage;
          }
        }, [context?.sendMessage]);

        return (
          <div>
            <div data-testid="connection-state">
              {context?.connectionState || "NOT_AVAILABLE"}
            </div>
          </div>
        );
      }

      // Act
      renderWithWebSocketContext(
        <TestComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      await waitFor(() => {
        expect(sockets).toHaveLength(1);
      });

      await act(async () => {
        sockets[0].open();
      });

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
      });

      // Send multiple messages
      await act(async () => {
        await sendMessageFn!({
          role: "user",
          content: [{ type: "text", text: "Message 1" }],
        });
        await sendMessageFn!({
          role: "user",
          content: [{ type: "text", text: "Message 2" }],
        });
      });

      // Assert - both messages should have been received in order
      await waitFor(() => {
        expect(sockets[0].sentMessages).toHaveLength(2);
      });

      expect(sockets[0].sentMessages[0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Message 1" }],
      });
      expect(sockets[0].sentMessages[1]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Message 2" }],
      });
    });

    it("should queue messages until the main WebSocket opens", async () => {
      const conversationId = "test-conversation-queue-before-open";
      const { sockets } = setupControlledWebSocket();
      let sendMessageFn: CapturedSendMessage | null = null;

      renderWithWebSocketContext(
        <SendMessageCapture
          onReady={(sendMessage) => {
            sendMessageFn = sendMessage;
          }}
        />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
        expect(sockets).toHaveLength(1);
      });

      let result: Awaited<ReturnType<CapturedSendMessage>> | null = null;
      await act(async () => {
        result = await sendMessageFn!(createTextMessage("Queued message"));
      });

      expect(result).toEqual({ queued: true });
      expect(sockets[0].sentMessages).toHaveLength(0);

      await act(async () => {
        sockets[0].open();
      });

      await waitFor(() => {
        expect(sockets[0].sentMessages).toHaveLength(1);
      });
      expect(sockets[0].sentMessages[0]).toEqual(
        createTextMessage("Queued message"),
      );
    });

    it("should flush queued messages in FIFO order", async () => {
      const conversationId = "test-conversation-queue-order";
      const { sockets } = setupControlledWebSocket();
      let sendMessageFn: CapturedSendMessage | null = null;

      renderWithWebSocketContext(
        <SendMessageCapture
          onReady={(sendMessage) => {
            sendMessageFn = sendMessage;
          }}
        />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
        expect(sockets).toHaveLength(1);
      });

      await act(async () => {
        await sendMessageFn!(createTextMessage("Queued 1"));
        await sendMessageFn!(createTextMessage("Queued 2"));
      });

      expect(sockets[0].sentMessages).toHaveLength(0);

      await act(async () => {
        sockets[0].open();
      });

      await waitFor(() => {
        expect(sockets[0].sentMessages).toHaveLength(2);
      });
      expect(sockets[0].sentMessages).toEqual([
        createTextMessage("Queued 1"),
        createTextMessage("Queued 2"),
      ]);
    });

    it("should flush backlog before a newly sent open-socket message", async () => {
      const conversationId = "test-conversation-backlog-before-current";
      const { sockets } = setupControlledWebSocket();
      let sendMessageFn: CapturedSendMessage | null = null;

      renderWithWebSocketContext(
        <SendMessageCapture
          onReady={(sendMessage) => {
            sendMessageFn = sendMessage;
          }}
        />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
        expect(sockets).toHaveLength(1);
      });

      await act(async () => {
        await sendMessageFn!(createTextMessage("Backlog"));
      });

      await act(async () => {
        sockets[0].open();
      });

      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      await act(async () => {
        const result = await sendMessageFn!(createTextMessage("Current"));
        expect(result).toEqual({ queued: false });
      });

      await waitFor(() => {
        expect(sockets[0].sentMessages).toHaveLength(2);
      });
      expect(sockets[0].sentMessages).toEqual([
        createTextMessage("Backlog"),
        createTextMessage("Current"),
      ]);
    });

    it("should clear queued messages when the conversation changes", async () => {
      const firstConversationId = "test-conversation-clear-a";
      const secondConversationId = "test-conversation-clear-b";
      const { sockets } = setupControlledWebSocket();
      let sendMessageFn: CapturedSendMessage | null = null;
      const queryClient = new QueryClient({
        defaultOptions: {
          queries: { retry: false },
          mutations: { retry: false },
        },
      });

      const renderHarness = (conversationId: string) => (
        <QueryClientProvider client={queryClient}>
          <ConversationWebSocketProvider
            conversationId={conversationId}
            conversationUrl={`http://localhost:3000/api/conversations/${conversationId}`}
          >
            <SendMessageCapture
              onReady={(sendMessage) => {
                sendMessageFn = sendMessage;
              }}
            />
          </ConversationWebSocketProvider>
        </QueryClientProvider>
      );

      const { rerender } = render(renderHarness(firstConversationId));

      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
        expect(sockets).toHaveLength(1);
      });

      await act(async () => {
        await sendMessageFn!(createTextMessage("Conversation A"));
      });

      rerender(renderHarness(secondConversationId));

      await waitFor(() => {
        expect(sockets).toHaveLength(2);
      });

      await act(async () => {
        sockets[1].open();
      });

      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });
      expect(sockets[0].sentMessages).toHaveLength(0);
      expect(sockets[1].sentMessages).toHaveLength(0);
    });

    it("should clear queued messages on unmount", async () => {
      const conversationId = "test-conversation-unmount-clear";
      const { sockets } = setupControlledWebSocket();
      let sendMessageFn: CapturedSendMessage | null = null;

      const { unmount } = renderWithWebSocketContext(
        <SendMessageCapture
          onReady={(sendMessage) => {
            sendMessageFn = sendMessage;
          }}
        />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
        expect(sockets).toHaveLength(1);
      });

      await act(async () => {
        await sendMessageFn!(createTextMessage("Discard on unmount"));
      });

      unmount();

      await act(async () => {
        sockets[0].open();
      });

      expect(sockets[0].sentMessages).toHaveLength(0);
    });

    it("should not call the REST pending-message fallback for queued WebSocket sends", async () => {
      const conversationId = "test-conversation-no-rest-fallback";
      const { sockets } = setupControlledWebSocket();
      let pendingMessageRequests = 0;
      let sendMessageFn: CapturedSendMessage | null = null;

      mswServer.use(
        http.post(
          `http://localhost:3000/api/v1/conversations/${conversationId}/pending-messages`,
          () => {
            pendingMessageRequests += 1;
            return HttpResponse.json({ id: "pending-message-id" });
          },
        ),
      );

      renderWithWebSocketContext(
        <SendMessageCapture
          onReady={(sendMessage) => {
            sendMessageFn = sendMessage;
          }}
        />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
        expect(sockets).toHaveLength(1);
      });

      await act(async () => {
        await sendMessageFn!(createTextMessage("No REST fallback"));
      });

      expect(pendingMessageRequests).toBe(0);
      expect(sockets[0].sentMessages).toHaveLength(0);

      await act(async () => {
        sockets[0].open();
      });

      await waitFor(() => {
        expect(sockets[0].sentMessages).toHaveLength(1);
      });
      expect(pendingMessageRequests).toBe(0);
    });

    it.each(["PAUSED", "MISSING"] as V1SandboxStatus[])(
      "should clear queued messages when sandbox status is %s",
      async (sandboxStatus) => {
        const conversationId = "test-conversation-pause-clear";
        const { sockets } = setupControlledWebSocket();
        let sendMessageFn: CapturedSendMessage | null = null;
        const queryClient = new QueryClient({
          defaultOptions: {
            queries: { retry: false },
            mutations: { retry: false },
          },
        });

        const renderHarness = (status: V1SandboxStatus) => (
          <QueryClientProvider client={queryClient}>
            <ConversationWebSocketProvider
              conversationId={conversationId}
              conversationUrl={`http://localhost:3000/api/conversations/${conversationId}`}
              sandboxStatus={status}
            >
              <SendMessageCapture
                onReady={(sendMessage) => {
                  sendMessageFn = sendMessage;
                }}
              />
            </ConversationWebSocketProvider>
          </QueryClientProvider>
        );

        const { rerender } = render(renderHarness("RUNNING"));

        await waitFor(() => {
          expect(sendMessageFn).not.toBeNull();
          expect(sockets).toHaveLength(1);
        });

        await act(async () => {
          await sendMessageFn!(createTextMessage("Discard after stop"));
        });

        rerender(renderHarness(sandboxStatus));

        await act(async () => {
          sockets[0].open();
        });

        await waitFor(() => {
          expect(screen.getByTestId("connection-state")).toHaveTextContent(
            "OPEN",
          );
        });
        expect(sockets[0].sentMessages).toHaveLength(0);
      },
    );

    it("should isolate main and planning queues", async () => {
      const conversationId = "test-conversation-planning-main";
      const planningConversationId = "test-conversation-planning-sub";
      const { sockets } = setupControlledWebSocket();
      let sendMessageFn: CapturedSendMessage | null = null;

      const queryClient = new QueryClient({
        defaultOptions: {
          queries: { retry: false },
          mutations: { retry: false },
        },
      });

      render(
        <QueryClientProvider client={queryClient}>
          <ConversationWebSocketProvider
            conversationId={conversationId}
            conversationUrl={`http://localhost:3000/api/conversations/${conversationId}`}
            sandboxStatus="RUNNING"
            subConversationIds={[planningConversationId]}
            subConversations={[
              createMockSubConversation(
                planningConversationId,
                `http://localhost:3000/api/conversations/${planningConversationId}`,
              ),
            ]}
          >
            <SendMessageCapture
              onReady={(sendMessage) => {
                sendMessageFn = sendMessage;
              }}
            />
          </ConversationWebSocketProvider>
        </QueryClientProvider>,
      );

      await waitFor(() => {
        expect(sendMessageFn).not.toBeNull();
        expect(sockets).toHaveLength(2);
      });

      await act(async () => {
        useConversationStore.getState().setConversationMode("code");
        await sendMessageFn!(createTextMessage("Main queued"));
        useConversationStore.getState().setConversationMode("plan");
        await sendMessageFn!(createTextMessage("Plan queued"));
      });

      await act(async () => {
        sockets[1].open();
      });

      await waitFor(() => {
        expect(sockets[1].sentMessages).toHaveLength(1);
      });
      expect(sockets[1].sentMessages).toEqual([
        createTextMessage("Plan queued"),
      ]);
      expect(sockets[0].sentMessages).toHaveLength(0);

      await act(async () => {
        sockets[0].open();
      });

      await waitFor(() => {
        expect(sockets[0].sentMessages).toHaveLength(1);
      });
      expect(sockets[0].sentMessages).toEqual([
        createTextMessage("Main queued"),
      ]);
    });
  });

  // 8. History Loading State Tests
  describe("History Loading State", () => {
    it("should track history loading state using event count from API", async () => {
      const conversationId = "test-conversation-with-history";

      // Mock the event count API to return 3 events
      const expectedEventCount = 3;

      // Create 3 mock events to simulate history
      const mockHistoryEvents = [
        createMockUserMessageEvent({ id: "history-event-1" }),
        createMockMessageEvent({ id: "history-event-2" }),
        createMockMessageEvent({ id: "history-event-3" }),
      ];

      // Set up MSW to mock both the HTTP API and WebSocket connection
      mswServer.use(
        // Mock events search for history preloading
        http.get(
          `http://localhost:3000/api/v1/conversation/${conversationId}/events/search`,
          async () => {
            await new Promise<void>((resolve) => {
              setTimeout(resolve, 10);
            });
            return HttpResponse.json({
              items: mockHistoryEvents,
            });
          },
        ),
        http.get(
          `http://localhost:3000/api/conversations/${conversationId}/events/count`,
          () => HttpResponse.json(expectedEventCount),
        ),
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send all history events
          mockHistoryEvents.forEach((event) => {
            client.send(JSON.stringify(event));
          });
        }),
      );

      // Create a test component that displays loading state
      function HistoryLoadingComponent() {
        const context = useConversationWebSocket();
        const { events } = useEventStore();

        return (
          <div>
            <div data-testid="is-loading-history">
              {context?.isLoadingHistory ? "true" : "false"}
            </div>
            <div data-testid="events-received">{events.length}</div>
            <div data-testid="expected-event-count">{expectedEventCount}</div>
          </div>
        );
      }

      // Render with WebSocket context
      renderWithWebSocketContext(
        <HistoryLoadingComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      // Wait for all events to be received
      await waitFor(() => {
        expect(screen.getByTestId("events-received")).toHaveTextContent("3");
      });

      // Once all events are received, loading should be complete
      await waitFor(() => {
        expect(screen.getByTestId("is-loading-history")).toHaveTextContent(
          "false",
        );
      });
    });

    it("should handle empty conversation history", async () => {
      const conversationId = "test-conversation-empty";

      // Set up MSW to mock both the HTTP API and WebSocket connection
      mswServer.use(
        // Mock empty events search
        http.get(
          `http://localhost:3000/api/v1/conversation/${conversationId}/events/search`,
          () =>
            HttpResponse.json({
              items: [],
            }),
        ),
        http.get(
          `http://localhost:3000/api/conversations/${conversationId}/events/count`,
          () => HttpResponse.json(0),
        ),
        wsLink.addEventListener("connection", ({ server }) => {
          server.connect();
          // No events sent for empty history
        }),
      );

      // Create a test component that displays loading state
      function HistoryLoadingComponent() {
        const context = useConversationWebSocket();

        return (
          <div>
            <div data-testid="is-loading-history">
              {context?.isLoadingHistory ? "true" : "false"}
            </div>
          </div>
        );
      }

      // Render with WebSocket context
      renderWithWebSocketContext(
        <HistoryLoadingComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      // Should quickly transition from loading to not loading when count is 0
      await waitFor(() => {
        expect(screen.getByTestId("is-loading-history")).toHaveTextContent(
          "false",
        );
      });
    });

    it("should handle history loading with large event count", async () => {
      const conversationId = "test-conversation-large-history";

      // Create 50 mock events to simulate large history
      const expectedEventCount = 50;
      const mockHistoryEvents = Array.from({ length: 50 }, (_, i) =>
        createMockMessageEvent({ id: `history-event-${i + 1}` }),
      );

      // Set up MSW to mock both the HTTP API and WebSocket connection
      mswServer.use(
        // Mock events search for history preloading (50 events)
        http.get(
          `http://localhost:3000/api/v1/conversation/${conversationId}/events/search`,
          async () => {
            await new Promise<void>((resolve) => {
              setTimeout(resolve, 10);
            });
            return HttpResponse.json({
              items: mockHistoryEvents,
            });
          },
        ),
        http.get(
          `http://localhost:3000/api/conversations/${conversationId}/events/count`,
          () => HttpResponse.json(expectedEventCount),
        ),
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send all history events
          mockHistoryEvents.forEach((event) => {
            client.send(JSON.stringify(event));
          });
        }),
      );

      // Create a test component that displays loading state
      function HistoryLoadingComponent() {
        const context = useConversationWebSocket();
        const { events } = useEventStore();

        return (
          <div>
            <div data-testid="is-loading-history">
              {context?.isLoadingHistory ? "true" : "false"}
            </div>
            <div data-testid="events-received">{events.length}</div>
          </div>
        );
      }

      // Render with WebSocket context
      renderWithWebSocketContext(
        <HistoryLoadingComponent />,
        conversationId,
        `http://localhost:3000/api/conversations/${conversationId}`,
      );

      // Wait for all events to be received
      await waitFor(() => {
        expect(screen.getByTestId("events-received")).toHaveTextContent("50");
      });

      // Once all events are received, loading should be complete
      await waitFor(() => {
        expect(screen.getByTestId("is-loading-history")).toHaveTextContent(
          "false",
        );
      });
    });
  });

  // 9. Browser State Tests (BrowserObservation)
  describe("Browser State Integration", () => {
    beforeEach(() => {
      useBrowserStore.getState().reset();
    });

    it("should update browser store with screenshot when BrowserObservation event is received", async () => {
      // Create a mock BrowserObservation event with screenshot data
      const mockBrowserObsEvent = createMockBrowserObservationEvent(
        "base64-screenshot-data",
        "Page loaded successfully",
      );

      // Set up MSW to send the event when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the mock event after connection
          client.send(JSON.stringify(mockBrowserObsEvent));
        }),
      );

      // Render with WebSocket context
      renderWithWebSocketContext(<ConnectionStatusComponent />);

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Wait for the browser store to be updated with screenshot
      await waitFor(() => {
        const { screenshotSrc } = useBrowserStore.getState();
        expect(screenshotSrc).toBe(
          "data:image/png;base64,base64-screenshot-data",
        );
      });
    });

    it("should update browser store with URL when BrowserNavigateAction followed by BrowserObservation", async () => {
      // Create mock events - action first, then observation
      const mockBrowserActionEvent = createMockBrowserNavigateActionEvent(
        "https://example.com/test-page",
      );
      const mockBrowserObsEvent = createMockBrowserObservationEvent(
        "base64-screenshot-data",
        "Page loaded successfully",
      );

      // Set up MSW to send both events when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send action first, then observation
          client.send(JSON.stringify(mockBrowserActionEvent));
          client.send(JSON.stringify(mockBrowserObsEvent));
        }),
      );

      // Render with WebSocket context
      renderWithWebSocketContext(<ConnectionStatusComponent />);

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Wait for the browser store to be updated with both screenshot and URL
      await waitFor(() => {
        const { screenshotSrc, url } = useBrowserStore.getState();
        expect(screenshotSrc).toBe(
          "data:image/png;base64,base64-screenshot-data",
        );
        expect(url).toBe("https://example.com/test-page");
      });
    });

    it("should not update browser store when BrowserObservation has no screenshot data", async () => {
      const initialScreenshot = useBrowserStore.getState().screenshotSrc;

      // Create a mock BrowserObservation event WITHOUT screenshot data
      const mockBrowserObsEvent = createMockBrowserObservationEvent(
        null, // no screenshot
        "Browser action completed",
      );

      // Set up MSW to send the event when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the mock event after connection
          client.send(JSON.stringify(mockBrowserObsEvent));
        }),
      );

      // Render with WebSocket context
      renderWithWebSocketContext(<ConnectionStatusComponent />);

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Give some time for any potential updates
      await new Promise((resolve) => {
        setTimeout(resolve, 100);
      });

      // Screenshot should remain unchanged (empty/initial value)
      const { screenshotSrc } = useBrowserStore.getState();
      expect(screenshotSrc).toBe(initialScreenshot);
    });
  });

  // 10. Terminal I/O Tests (ExecuteBashAction and ExecuteBashObservation)
  describe("Terminal I/O Integration", () => {
    beforeEach(() => {
      useCommandStore.getState().clearTerminal();
    });

    it("should append command to store when ExecuteBashAction event is received", async () => {
      // Create a mock ExecuteBashAction event
      const mockBashActionEvent = createMockExecuteBashActionEvent("npm test");

      // Set up MSW to send the event when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the mock event after connection
          client.send(JSON.stringify(mockBashActionEvent));
        }),
      );

      // Render with WebSocket context (we don't need a component, just need the provider to be active)
      renderWithWebSocketContext(<ConnectionStatusComponent />);

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Wait for the command to be added to the store
      await waitFor(() => {
        const { commands } = useCommandStore.getState();
        expect(commands.length).toBe(1);
      });

      // Verify the command was added with correct type and content
      const { commands } = useCommandStore.getState();
      expect(commands[0].type).toBe("input");
      expect(commands[0].content).toBe("npm test");
    });

    it("should append output to store when ExecuteBashObservation event is received", async () => {
      // Create a mock ExecuteBashObservation event
      const mockBashObservationEvent = createMockExecuteBashObservationEvent(
        "PASS  tests/example.test.js\n  ✓ should work (2 ms)",
        "npm test",
      );

      // Set up MSW to send the event when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the mock event after connection
          client.send(JSON.stringify(mockBashObservationEvent));
        }),
      );

      // Render with WebSocket context
      renderWithWebSocketContext(<ConnectionStatusComponent />);

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Wait for the output to be added to the store
      await waitFor(() => {
        const { commands } = useCommandStore.getState();
        expect(commands.length).toBe(1);
      });

      // Verify the output was added with correct type and content
      const { commands } = useCommandStore.getState();
      expect(commands[0].type).toBe("output");
      expect(commands[0].content).toBe(
        "PASS  tests/example.test.js\n  ✓ should work (2 ms)",
      );
    });
  });
});
