import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
  afterEach,
  vi,
} from "vitest";
import { screen, waitFor, render, cleanup } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createMockAgentErrorEvent,
  createMockConversationErrorEvent,
} from "#/mocks/mock-ws-helpers";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { conversationWebSocketTestSetup } from "./helpers/msw-websocket-setup";
import { ConnectionStatusComponent } from "./helpers/websocket-test-components";

// Mock the tracking function
const mockTrackCreditLimitReached = vi.fn();

// Mock useTracking hook
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackCreditLimitReached: mockTrackCreditLimitReached,
    trackLoginButtonClick: vi.fn(),
    trackConversationCreated: vi.fn(),
    trackPushButtonClick: vi.fn(),
    trackPullButtonClick: vi.fn(),
    trackCreatePrButtonClick: vi.fn(),
    trackGitProviderConnected: vi.fn(),
    trackUserSignupCompleted: vi.fn(),
    trackCreditsPurchased: vi.fn(),
  }),
}));

// Mock useActiveConversation hook
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: null,
    isLoading: false,
    error: null,
  }),
}));

// MSW WebSocket mock setup
const { wsLink, server: mswServer } = conversationWebSocketTestSetup();

beforeAll(() => {
  // The global MSW server from vitest.setup.ts is already running
  // We just need to start our WebSocket-specific server
  mswServer.listen({ onUnhandledRequest: "bypass" });
});

afterEach(() => {
  // Clear all mocks before each test
  mockTrackCreditLimitReached.mockClear();
  mswServer.resetHandlers();
  // Clean up any React components
  cleanup();
});

afterAll(async () => {
  // Close the WebSocket MSW server
  mswServer.close();

  // Give time for any pending WebSocket connections to close. This is very important to prevent serious memory leaks
  await new Promise((resolve) => {
    setTimeout(resolve, 500);
  });
});

// Helper function to render components with all necessary providers
function renderWithProviders(
  children: React.ReactNode,
  conversationId = "test-conversation-123",
  conversationUrl = "http://localhost:3000/api/conversations/test-conversation-123",
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <ConversationWebSocketProvider
        conversationId={conversationId}
        conversationUrl={conversationUrl}
        sessionApiKey={null}
      >
        {children}
      </ConversationWebSocketProvider>
    </QueryClientProvider>,
  );
}

describe("PostHog Analytics Tracking", () => {
  describe("Credit Limit Tracking", () => {
    it("should track credit_limit_reached when AgentErrorEvent contains budget error", async () => {
      // Create a mock AgentErrorEvent with budget-related error message
      const mockBudgetErrorEvent = createMockAgentErrorEvent({
        error: "ExceededBudget: Task exceeded maximum budget of $10.00",
      });

      // Set up MSW to send the budget error event when connection is established
      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the mock budget error event after connection
          client.send(JSON.stringify(mockBudgetErrorEvent));
        }),
      );

      // Render with all providers
      renderWithProviders(<ConnectionStatusComponent />);

      // Wait for connection to be established
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Wait for the tracking event to be captured
      await waitFor(() => {
        expect(mockTrackCreditLimitReached).toHaveBeenCalledWith(
          expect.objectContaining({
            conversationId: "test-conversation-123",
          }),
        );
      });
    });

    it("should track credit_limit_reached when AgentErrorEvent contains 'credit' keyword", async () => {
      // Create error with "credit" keyword (case-insensitive)
      const mockCreditErrorEvent = createMockAgentErrorEvent({
        error: "Insufficient CREDIT to complete this operation",
      });

      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          client.send(JSON.stringify(mockCreditErrorEvent));
        }),
      );

      renderWithProviders(<ConnectionStatusComponent />);

      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      await waitFor(() => {
        expect(mockTrackCreditLimitReached).toHaveBeenCalledWith(
          expect.objectContaining({
            conversationId: "test-conversation-123",
          }),
        );
      });
    });

    it("should NOT track credit_limit_reached for non-budget errors", async () => {
      // Create a regular error without budget/credit keywords
      const mockRegularErrorEvent = createMockAgentErrorEvent({
        error: "Failed to execute command: Permission denied",
      });

      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          client.send(JSON.stringify(mockRegularErrorEvent));
        }),
      );

      renderWithProviders(<ConnectionStatusComponent />);

      // Wait for connection and error to be processed
      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      // Verify that credit_limit_reached was NOT tracked
      expect(mockTrackCreditLimitReached).not.toHaveBeenCalled();
    });

    it("should only track credit_limit_reached once per error event", async () => {
      const mockBudgetErrorEvent = createMockAgentErrorEvent({
        error: "Budget exceeded: $10.00 limit reached",
      });

      mswServer.use(
        wsLink.addEventListener("connection", ({ client, server }) => {
          server.connect();
          // Send the same error event twice
          client.send(JSON.stringify(mockBudgetErrorEvent));
          client.send(
            JSON.stringify({ ...mockBudgetErrorEvent, id: "different-id" }),
          );
        }),
      );

      renderWithProviders(<ConnectionStatusComponent />);

      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      await waitFor(() => {
        expect(mockTrackCreditLimitReached).toHaveBeenCalledTimes(2);
      });

      // Both calls should be for credit_limit_reached (once per event)
      expect(mockTrackCreditLimitReached).toHaveBeenNthCalledWith(
        1,
        expect.objectContaining({
          conversationId: "test-conversation-123",
        }),
      );
      expect(mockTrackCreditLimitReached).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          conversationId: "test-conversation-123",
        }),
      );
    });

    it("should track credit_limit_reached when ConversationErrorEvent contains budget error", async () => {
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

      renderWithProviders(<ConnectionStatusComponent />);

      await waitFor(() => {
        expect(screen.getByTestId("connection-state")).toHaveTextContent(
          "OPEN",
        );
      });

      await waitFor(() => {
        expect(mockTrackCreditLimitReached).toHaveBeenCalledWith(
          expect.objectContaining({
            conversationId: "test-conversation-123",
          }),
        );
      });
    });
  });
});
