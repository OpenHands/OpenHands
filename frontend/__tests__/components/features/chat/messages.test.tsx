import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Messages } from "#/components/features/chat/messages";
import {
  AssistantMessageAction,
  OpenHandsAction,
  UserMessageAction,
} from "#/types/core/actions";
import { OpenHandsObservation } from "#/types/core/observations";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { Conversation } from "#/api/open-hands.types";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

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
    const getConversationSpy = vi.spyOn(ConversationService, "getConversation");
    const mockConversation: Conversation = {
      conversation_id: "123",
      title: "Test Conversation",
      status: "RUNNING",
      runtime_status: "STATUS$READY",
      created_at: new Date().toISOString(),
      last_updated_at: new Date().toISOString(),
      selected_branch: null,
      selected_repository: null,
      git_provider: "github",
      session_api_key: null,
      url: null,
    };

    getConversationSpy.mockResolvedValue(mockConversation);

    renderMessages({
      messages: [userMessage, assistantMessage],
    });

    expect(screen.getByText("Hello, User!")).toBeInTheDocument();
    expect(screen.getByText("Hello, Assistant!")).toBeInTheDocument();
  });
});
