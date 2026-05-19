import React from "react";
import { describe, it, expect, vi, beforeEach, afterAll } from "vitest";
import { waitFor, render, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router";
import {
  ConversationWebSocketProvider,
  useConversationWebSocket,
} from "#/contexts/conversation-websocket-context";

const mockSocketData = vi.hoisted(() => {
  const mockSocket: {
    readyState: number;
    send: ReturnType<typeof vi.fn>;
  } = {
    readyState: WebSocket.CONNECTING,
    send: vi.fn(),
  };

  return {
    mockSocket,
    useWebSocket: vi.fn(() => ({
      socket: mockSocket as unknown as WebSocket,
      isConnected: false,
      lastMessage: null,
      messages: [],
      error: null,
      sendMessage: vi.fn(),
      isReconnecting: false,
      attemptCount: 0,
      disconnect: vi.fn(),
    })),
  };
});

vi.mock("#/hooks/use-websocket", () => ({
  useWebSocket: mockSocketData.useWebSocket,
}));

afterAll(() => {
  vi.restoreAllMocks();
});

function buildTestTree(
  children: React.ReactNode,
  conversationId = "test-conversation-default",
  conversationUrl = "http://localhost:3000/api/conversations/test-conversation-default",
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/test-conversation-default"]}>
        <Routes>
          <Route
            path="/:conversationId"
            element={
              <ConversationWebSocketProvider
                conversationId={conversationId}
                conversationUrl={conversationUrl}
              >
                {children}
              </ConversationWebSocketProvider>
            }
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

function renderWithWebSocketContext(
  children: React.ReactNode,
  conversationId?: string,
  conversationUrl?: string,
) {
  return render(buildTestTree(children, conversationId, conversationUrl));
}

describe("WebSocket Message Queuing", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSocketData.mockSocket.readyState = WebSocket.CONNECTING;
    mockSocketData.useWebSocket.mockReturnValue({
      socket: mockSocketData.mockSocket as unknown as WebSocket,
      isConnected: false,
      lastMessage: null,
      messages: [],
      error: null,
      sendMessage: vi.fn(),
      isReconnecting: false,
      attemptCount: 0,
      disconnect: vi.fn(),
    });
  });

  it("should queue messages in memory when WebSocket is CONNECTING", async () => {
    let capturedSendMessage:
      | NonNullable<ReturnType<typeof useConversationWebSocket>>["sendMessage"]
      | null = null;

    function TestComponent() {
      const ctx = useConversationWebSocket();
      React.useEffect(() => {
        if (ctx?.sendMessage) capturedSendMessage = ctx.sendMessage;
      }, [ctx?.sendMessage]);
      return null;
    }

    renderWithWebSocketContext(<TestComponent />);
    await waitFor(() => expect(capturedSendMessage).not.toBeNull());

    const result = await act(async () =>
      capturedSendMessage!({
        role: "user",
        content: [{ type: "text", text: "Queued" }],
      }),
    );

    expect(result).toEqual({ queued: true });
  });

  it("should flush queued messages when WebSocket transitions to OPEN", async () => {
    let capturedSendMessage:
      | NonNullable<ReturnType<typeof useConversationWebSocket>>["sendMessage"]
      | null = null;

    function InnerComponent() {
      const ctx = useConversationWebSocket();
      React.useEffect(() => {
        if (ctx?.sendMessage) capturedSendMessage = ctx.sendMessage;
      }, [ctx?.sendMessage]);
      return null;
    }

    const { rerender } = renderWithWebSocketContext(<InnerComponent />);
    await waitFor(() => expect(capturedSendMessage).not.toBeNull());

    // Queue a message while socket is CONNECTING
    await act(async () => {
      await capturedSendMessage!({
        role: "user",
        content: [{ type: "text", text: "Flush me" }],
      });
    });

    expect(mockSocketData.mockSocket.send).not.toHaveBeenCalled();

    // Simulate socket opening — update both readyState and mock return value
    mockSocketData.mockSocket.readyState = WebSocket.OPEN;
    mockSocketData.useWebSocket.mockReturnValue({
      socket: mockSocketData.mockSocket as unknown as WebSocket,
      isConnected: true,
      lastMessage: null,
      messages: [],
      error: null,
      sendMessage: vi.fn(),
      isReconnecting: false,
      attemptCount: 0,
      disconnect: vi.fn(),
    });

    // Re-render to trigger the flush useEffect (depends on mainSocket?.readyState)
    rerender(buildTestTree(<InnerComponent />));

    await waitFor(() => {
      expect(mockSocketData.mockSocket.send).toHaveBeenCalledWith(
        JSON.stringify({
          role: "user",
          content: [{ type: "text", text: "Flush me" }],
        }),
      );
    });
  });
});
