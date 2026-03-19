import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  updateStatusWhenErrorMessagePresent,
  WsClientProvider,
  useWsClient,
} from "#/context/ws-client-provider";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

describe("Propagate error message", () => {
  it("should do nothing when no message was passed from server", () => {
    updateStatusWhenErrorMessagePresent(null);
    updateStatusWhenErrorMessagePresent(undefined);
    updateStatusWhenErrorMessagePresent({});
    updateStatusWhenErrorMessagePresent({ message: null });
  });

  it.todo("should display error to user when present");

  it.todo("should display error including translation id when present");
});

// Create a mock for socket.io-client
const mockEmit = vi.fn();
const mockOn = vi.fn();
const mockOff = vi.fn();
const mockDisconnect = vi.fn();

vi.mock("socket.io-client", () => ({
  io: vi.fn(() => ({
    emit: mockEmit,
    on: mockOn,
    off: mockOff,
    disconnect: mockDisconnect,
    io: {
      opts: {
        query: {},
      },
    },
  })),
}));

// Mock component to test the hook
function TestComponent() {
  const { send } = useWsClient();

  React.useEffect(() => {
    // Send a test event
    send({ type: "test_event" });
  }, [send]);

  return <div>Test Component</div>;
}

describe("WsClientProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useSelectedOrganizationStore.setState({ organizationId: "test-org-id" });
    vi.mock("#/hooks/query/use-active-conversation", () => ({
      useActiveConversation: () => {
        return { data: {
        conversation_id: "1",
        title: "Conversation 1",
        selected_repository: null,
        last_updated_at: "2021-10-01T12:00:00Z",
        created_at: "2021-10-01T12:00:00Z",
        status: "RUNNING" as const,
        runtime_status: "STATUS$READY",
        url: null,
        session_api_key: null,
      }}},
    }));
  });

  it("should emit oh_user_action event when send is called", async () => {
    const { getByText } = render(<TestComponent />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={new QueryClient()}>
          <WsClientProvider conversationId="test-conversation-id">
            {children}
          </WsClientProvider>
        </QueryClientProvider>
      ),
    });

    // Assert
    expect(getByText("Test Component")).toBeInTheDocument();

    // Wait for the emit call to happen (useEffect needs time to run)
    await waitFor(
      () => {
        expect(mockEmit).toHaveBeenCalledWith("oh_user_action", {
          type: "test_event",
        });
      },
      { timeout: 1000 },
    );
  });
});
