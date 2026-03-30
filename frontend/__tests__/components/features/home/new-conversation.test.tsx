import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { createRoutesStub } from "react-router";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { NewConversation } from "#/components/features/home/new-conversation/new-conversation";

vi.mock("#/hooks/query/use-settings", async () => {
  const actual = await vi.importActual<typeof import("#/hooks/query/use-settings")>(
    "#/hooks/query/use-settings",
  );
  return {
    ...actual,
    getSettingsQueryFn: vi.fn().mockResolvedValue({ v1_enabled: true }),
  };
});

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: () => ({ organizationId: null }),
}));

// Mock the translation function
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        // Return a mock translation for the test
        const translations: Record<string, string> = {
          COMMON$START_FROM_SCRATCH: "Start from Scratch",
          HOME$NEW_PROJECT_DESCRIPTION: "Create a new project from scratch",
          COMMON$NEW_CONVERSATION: "New Conversation",
          HOME$LOADING: "Loading...",
        };
        return translations[key] || key;
      },
      i18n: { language: "en" },
    }),
  };
});

const renderNewConversation = () => {
  const RouterStub = createRoutesStub([
    {
      Component: NewConversation,
      path: "/",
    },
    {
      Component: () => <div data-testid="conversation-screen" />,
      path: "/conversations/:conversationId",
    },
  ]);

  return render(<RouterStub />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });
};

describe("NewConversation", () => {
  it("should create an empty conversation and redirect when pressing the launch from scratch button", async () => {
    const createConversationSpy = vi
      .spyOn(V1ConversationService, "createConversation")
      .mockResolvedValue({
        id: "task-id",
        created_by_user_id: null,
        status: "READY",
        detail: null,
        app_conversation_id: "conv-123",
        sandbox_id: null,
        agent_server_url: "http://agent-server.local",
        request: {
          sandbox_id: null,
          initial_message: null,
          processors: [],
          llm_model: null,
          selected_repository: null,
          selected_branch: null,
          git_provider: "github",
          suggested_task: null,
          title: null,
          trigger: null,
          pr_number: [],
          parent_conversation_id: null,
          agent_type: "default",
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });

    renderNewConversation();

    const launchButton = screen.getByTestId("launch-new-conversation-button");
    await userEvent.click(launchButton);

    expect(createConversationSpy).toHaveBeenCalledOnce();

    // expect to be redirected to /conversations/:conversationId
    await screen.findByTestId("conversation-screen");
  });

  it("should change the launch button text to 'Loading...' when creating a conversation", async () => {
    // Mock V1 API to never resolve, keeping the mutation in loading state
    vi.spyOn(V1ConversationService, "createConversation").mockImplementation(
      () => new Promise(() => {}),
    );

    renderNewConversation();

    const launchButton = screen.getByTestId("launch-new-conversation-button");
    await userEvent.click(launchButton);

    expect(launchButton).toHaveTextContent(/Loading.../i);
    expect(launchButton).toBeDisabled();
  });
});
