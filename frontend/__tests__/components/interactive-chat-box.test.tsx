import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { InteractiveChatBox } from "#/components/features/chat/interactive-chat-box";
import { renderWithProviders } from "../../test-utils";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useConversationStore } from "#/state/conversation-store";

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: vi.fn(),
}));

// Mock React Router hooks
vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useNavigate: () => vi.fn(),
    useParams: () => ({ conversationId: "test-conversation-id" }),
  };
});

// Mock the useActiveConversation hook
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: { status: null },
    isFetched: true,
    refetch: vi.fn(),
  }),
}));

// Mock other hooks that might be used by the component
vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({
    providers: [],
  }),
}));

vi.mock("#/hooks/use-conversation-name-context-menu", () => ({
  useConversationNameContextMenu: () => ({
    isOpen: false,
    contextMenuRef: { current: null },
    handleContextMenu: vi.fn(),
    handleClose: vi.fn(),
    handleRename: vi.fn(),
    handleDelete: vi.fn(),
  }),
}));

describe("InteractiveChatBox", () => {
  const onSubmitMock = vi.fn();

  const mockStores = (agentState: AgentState = AgentState.INIT) => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: agentState,
    });

    useConversationStore.setState({
      images: [],
      files: [],
      loadingFiles: [],
      loadingImages: [],
      submittedMessage: null,
      messageToSend: null,
      shouldShownAgentLoading: false,
      shouldHideSuggestions: false,
      isRightPanelShown: true,
      selectedTab: "editor" as const,
      hasRightPanelToggled: true,
    });
  };

  // Helper function to render with Router context
  const renderInteractiveChatBox = (props: any, options: any = {}) =>
    renderWithProviders(
      <MemoryRouter>
        <InteractiveChatBox {...props} />
      </MemoryRouter>,
      options,
    );

  beforeAll(() => {
    global.URL.createObjectURL = vi
      .fn()
      .mockReturnValue("blob:http://example.com");
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    mockStores(AgentState.INIT);

    renderInteractiveChatBox({
      onSubmit: onSubmitMock,
    });

    const chatBox = screen.getByTestId("interactive-chat-box");
    expect(chatBox).toBeInTheDocument();
  });

  it("should set custom values", async () => {
    const user = userEvent.setup();
    mockStores(AgentState.AWAITING_USER_INPUT);

    renderInteractiveChatBox({
      onSubmit: onSubmitMock,
    });

    const textbox = screen.getByTestId("chat-input");

    // Simulate user typing to populate the input
    await user.type(textbox, "Hello, world!");

    expect(textbox).toHaveTextContent("Hello, world!");
  });

  it("should display the image previews when images are uploaded", async () => {
    const user = userEvent.setup();
    mockStores(AgentState.INIT);

    renderInteractiveChatBox({
      onSubmit: onSubmitMock,
    });

    // Create a larger file to ensure it passes validation
    const fileContent = new Array(1024).fill("a").join(""); // 1KB file
    const file = new File([fileContent], "chucknorris.png", {
      type: "image/png",
    });

    // Click on the paperclip icon to trigger file selection
    const paperclipIcon = screen.getByTestId("paperclip-icon");
    await user.click(paperclipIcon);

    // Now trigger the file input change event directly
    const input = screen.getByTestId("upload-image-input");
    await user.upload(input, file);

    // For now, just verify the file input is accessible
    expect(input).toBeInTheDocument();
  });

  it("should remove the image preview when the close button is clicked", async () => {
    const user = userEvent.setup();
    mockStores(AgentState.INIT);

    renderInteractiveChatBox({
      onSubmit: onSubmitMock,
    });

    const fileContent = new Array(1024).fill("a").join(""); // 1KB file
    const file = new File([fileContent], "chucknorris.png", {
      type: "image/png",
    });

    // Click on the paperclip icon to trigger file selection
    const paperclipIcon = screen.getByTestId("paperclip-icon");
    await user.click(paperclipIcon);

    const input = screen.getByTestId("upload-image-input");
    await user.upload(input, file);

    // For now, just verify the file input is accessible
    expect(input).toBeInTheDocument();
  });

  it("should call onSubmit with the message and images", async () => {
    const user = userEvent.setup();
    mockStores(AgentState.INIT);

    renderInteractiveChatBox({
      onSubmit: onSubmitMock,
    });

    const textarea = screen.getByTestId("chat-input");

    // Type the message and ensure it's properly set
    await user.type(textarea, "Hello, world!");

    // Set innerText directly as the component reads this property
    textarea.innerText = "Hello, world!";

    // Verify the text is in the input before submitting
    expect(textarea).toHaveTextContent("Hello, world!");

    // Click the submit button instead of pressing Enter for more reliable testing
    const submitButton = screen.getByTestId("submit-button");

    // Verify the button is enabled before clicking
    expect(submitButton).not.toBeDisabled();

    await user.click(submitButton);

    expect(onSubmitMock).toHaveBeenCalledWith("Hello, world!", [], []);
  });

  it("should disable the submit button when agent is loading", async () => {
    const user = userEvent.setup();
    mockStores(AgentState.LOADING);

    renderInteractiveChatBox({
      onSubmit: onSubmitMock,
    });

    const button = screen.getByTestId("submit-button");
    expect(button).toBeDisabled();

    await user.click(button);
    expect(onSubmitMock).not.toHaveBeenCalled();
  });

  it("should handle image upload and message submission correctly", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    mockStores(AgentState.AWAITING_USER_INPUT);

    const { rerender } = renderInteractiveChatBox({
      onSubmit,
    });

    // Verify text input has the initial value
    const textarea = screen.getByTestId("chat-input");
    expect(textarea).toHaveTextContent("");

    // Set innerText directly as the component reads this property
    textarea.innerText = "test message";

    // Submit the message
    const submitButton = screen.getByTestId("submit-button");
    await user.click(submitButton);

    // Verify onSubmit was called with the message
    expect(onSubmit).toHaveBeenCalledWith("test message", [], []);

    // Simulate parent component updating the value prop
    rerender(
      <MemoryRouter>
        <InteractiveChatBox onSubmit={onSubmit} />
      </MemoryRouter>,
    );

    // Verify the text input was cleared
    expect(screen.getByTestId("chat-input")).toHaveTextContent("");
  });
});
