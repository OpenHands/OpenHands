import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Messages } from "#/components/features/chat/messages";
import {
  AssistantMessageAction,
  OpenHandsAction,
  UserMessageAction,
} from "#/types/core/actions";
import { OpenHandsObservation } from "#/types/core/observations";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";

vi.mock("react-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-router")>()),
  useParams: () => ({ conversationId: "123" }),
  useRevalidator: () => ({ revalidate: vi.fn() }),
}));

let queryClient: QueryClient;

const renderMessages = ({
  messages,
}: {
  messages: (OpenHandsAction | OpenHandsObservation)[];
}) => {
  const { rerender, ...rest } = render(
    <Messages messages={messages} isAwaitingUserConfirmation={false} />,
    {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient!}>
          {children}
        </QueryClientProvider>
      ),
    },
  );

  const rerenderMessages = (
    newMessages: (OpenHandsAction | OpenHandsObservation)[],
  ) => {
    rerender(
      <Messages messages={newMessages} isAwaitingUserConfirmation={false} />,
    );
  };

  return { ...rest, rerender: rerenderMessages };
};

describe("Messages", () => {
  beforeEach(() => {
    queryClient = new QueryClient();
    useSelectedOrganizationStore.setState({ organizationId: "test-org-id" });
  });

  afterEach(() => {
    // Reset optimistic-user-message zustand store between tests so a pending
    // bubble from one case does not leak into the next (the store is a global
    // singleton).
    useOptimisticUserMessageStore.getState().removeOptimisticUserMessage();
  });

  const assistantMessage: AssistantMessageAction = {
    id: 0,
    action: "message",
    source: "agent",
    message: "Hello, Assistant!",
    timestamp: new Date().toISOString(),
    args: {
      image_urls: [],
      file_urls: [],
      thought: "",
      wait_for_response: false,
    },
  };

  const userMessage: UserMessageAction = {
    id: 1,
    action: "message",
    source: "user",
    message: "Hello, User!",
    timestamp: new Date().toISOString(),
    args: { content: "Hello, User!", image_urls: [], file_urls: [] },
  };

  it("should render", () => {
    renderMessages({ messages: [userMessage, assistantMessage] });

    expect(screen.getByText("Hello, User!")).toBeInTheDocument();
    expect(screen.getByText("Hello, Assistant!")).toBeInTheDocument();
  });

  it("should render a launch to microagent action button on chat messages only if it is a user message", () => {
    renderMessages({
      messages: [userMessage, assistantMessage],
    });

    expect(screen.getByText("Hello, User!")).toBeInTheDocument();
    expect(screen.getByText("Hello, Assistant!")).toBeInTheDocument();
  });

  // Issue #14181: when an optimistic user message has been queued via the
  // pending-message REST endpoint, the wrapper must thread the pending flag
  // from the store through to <ChatMessage> so the "Delivering..." indicator
  // renders. This guards the V0 wrapper end-to-end (store -> wrapper -> bubble),
  // complementing the prop-level cases in chat-message.test.tsx.
  it("should render the Delivering indicator on the optimistic bubble when the message is pending delivery", () => {
    useOptimisticUserMessageStore
      .getState()
      .setOptimisticUserMessage("Queued from REST", true);

    renderMessages({ messages: [] });

    const indicator = screen.getByTestId("delivering-indicator");
    expect(indicator).toBeInTheDocument();
    expect(indicator).toHaveAttribute("role", "status");
    expect(indicator).toHaveAttribute("aria-live", "polite");
    expect(screen.getByText("Queued from REST")).toBeInTheDocument();
  });

  it("should not render the Delivering indicator when the optimistic message is not pending", () => {
    useOptimisticUserMessageStore
      .getState()
      .setOptimisticUserMessage("Sent over WS", false);

    renderMessages({ messages: [] });

    expect(screen.getByText("Sent over WS")).toBeInTheDocument();
    expect(
      screen.queryByTestId("delivering-indicator"),
    ).not.toBeInTheDocument();
  });
});
