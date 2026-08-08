import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { NewConversation } from "#/components/features/home/new-conversation/new-conversation";

const { mockCreateConversationMutate, mockUseCreateConversation } = vi.hoisted(
  () => ({
    mockCreateConversationMutate: vi.fn(),
    mockUseCreateConversation: vi.fn(),
  }),
);

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => mockUseCreateConversation(),
}));

vi.mock("#/hooks/query/use-settings", async () => {
  const actual = await vi.importActual<typeof import("#/hooks/query/use-settings")>(
    "#/hooks/query/use-settings",
  );
  return {
    ...actual,
    getSettingsQueryFn: vi.fn().mockResolvedValue({}),
  };
});

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

const renderNewConversation = (navigate = vi.fn()) =>
  renderWithProviders(<NewConversation />, {
    navigation: { navigate },
  });

describe("NewConversation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCreateConversationMutate.mockImplementation((_variables, options) => {
      options?.onSuccess?.({
        conversation_id: "conv-123",
        session_api_key: null,
        url: null,
      });
    });
    mockUseCreateConversation.mockReturnValue({
      mutate: mockCreateConversationMutate,
      isPending: false,
      isSuccess: false,
    });
  });

  it("should create an empty conversation and navigate when pressing the launch from scratch button", async () => {
    const navigate = vi.fn();

    renderNewConversation(navigate);

    const launchButton = screen.getByTestId("launch-new-conversation-button");
    await userEvent.click(launchButton);

    expect(mockCreateConversationMutate).toHaveBeenCalledOnce();
    expect(mockCreateConversationMutate).toHaveBeenCalledWith(
      { entryPoint: "home_new_conversation_button" },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith("/conversations/conv-123");
    });
  });

  it("should change the launch button text to 'Loading...' while creating a conversation", () => {
    mockUseCreateConversation.mockReturnValue({
      mutate: mockCreateConversationMutate,
      isPending: true,
      isSuccess: false,
    });

    renderNewConversation();

    const launchButton = screen.getByTestId("launch-new-conversation-button");

    expect(launchButton).toHaveTextContent(/Loading.../i);
    expect(launchButton).toBeDisabled();
  });
});
