import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { PlanPreview } from "#/components/features/chat/plan-preview";
import { useConversationStore } from "#/stores/conversation-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { createChatMessage } from "#/services/chat-service";

// Mock the feature flag to always return true (not testing feature flag behavior)
vi.mock("#/utils/feature-flags", () => ({
  USE_PLANNING_AGENT: vi.fn(() => true),
}));

// Mock i18n - need to preserve initReactI18next and I18nextProvider for test-utils
vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => key,
    }),
  };
});

// Mock services (underlying dependencies of the hook)
const mockSend = vi.fn();

vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: vi.fn(() => ({
    send: mockSend,
  })),
}));

vi.mock("#/services/chat-service", () => ({
  createChatMessage: vi.fn((content, imageUrls, fileUrls, timestamp) => ({
    action: "message",
    args: { content, image_urls: imageUrls, file_urls: fileUrls, timestamp },
  })),
}));

describe("PlanPreview", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Reset store states
    useConversationStore.setState({
      conversationMode: "plan",
    });
    useOptimisticUserMessageStore.setState({
      optimisticUserMessage: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();

    // Clean up store states
    useConversationStore.setState({
      conversationMode: "code",
    });
    useOptimisticUserMessageStore.setState({
      optimisticUserMessage: null,
    });
  });

  it("should render nothing when planContent is null", () => {
    renderWithProviders(<PlanPreview planContent={null} />);

    const contentDiv = screen.getByTestId("plan-preview-content");
    expect(contentDiv).toBeInTheDocument();
    expect(contentDiv.textContent?.trim() || "").toBe("");
  });

  it("should render nothing when planContent is undefined", () => {
    renderWithProviders(<PlanPreview planContent={undefined} />);

    const contentDiv = screen.getByTestId("plan-preview-content");
    expect(contentDiv).toBeInTheDocument();
    expect(contentDiv.textContent?.trim() || "").toBe("");
  });

  it("should render markdown content when planContent is provided", () => {
    const planContent = "# Plan Title\n\nThis is the plan content.";

    const { container } = renderWithProviders(
      <PlanPreview planContent={planContent} />,
    );

    // Check that component rendered and contains the content (markdown may break up text)
    expect(container.firstChild).not.toBeNull();
    expect(container.textContent).toContain("Plan Title");
    expect(container.textContent).toContain("This is the plan content.");
  });

  it("should render full content when length is less than or equal to 300 characters", () => {
    const planContent = "A".repeat(300);

    const { container } = renderWithProviders(
      <PlanPreview planContent={planContent} />,
    );

    // Content should be present (may be broken up by markdown)
    expect(container.textContent).toContain(planContent);
    expect(screen.queryByText(/COMMON\$READ_MORE/i)).not.toBeInTheDocument();
  });

  it("should truncate content when length exceeds 300 characters", () => {
    const longContent = "A".repeat(350);

    const { container } = renderWithProviders(
      <PlanPreview planContent={longContent} />,
    );

    // Truncated content should be present (may be broken up by markdown)
    expect(container.textContent).toContain("A".repeat(300));
    expect(container.textContent).toContain("...");
    expect(container.textContent).toContain("COMMON$READ_MORE");
  });

  it("should call onViewClick when View button is clicked", async () => {
    const user = userEvent.setup();
    const onViewClick = vi.fn();

    renderWithProviders(
      <PlanPreview planContent="Plan content" onViewClick={onViewClick} />,
    );

    const viewButton = screen.getByTestId("plan-preview-view-button");
    expect(viewButton).toBeInTheDocument();

    await user.click(viewButton);

    expect(onViewClick).toHaveBeenCalledTimes(1);
  });

  it("should call onViewClick when Read More button is clicked", async () => {
    const user = userEvent.setup();
    const onViewClick = vi.fn();
    const longContent = "A".repeat(350);

    renderWithProviders(
      <PlanPreview planContent={longContent} onViewClick={onViewClick} />,
    );

    const readMoreButton = screen.getByTestId("plan-preview-read-more-button");
    expect(readMoreButton).toBeInTheDocument();

    await user.click(readMoreButton);

    expect(onViewClick).toHaveBeenCalledTimes(1);
  });

  it("should render Build button", () => {
    renderWithProviders(<PlanPreview planContent="Plan content" />);

    const buildButton = screen.getByTestId("plan-preview-build-button");
    expect(buildButton).toBeInTheDocument();
  });

  it("should switch to code mode when Build button is clicked", async () => {
    // Arrange
    useConversationStore.setState({ conversationMode: "plan" });
    const user = userEvent.setup();
    renderWithProviders(<PlanPreview planContent="Plan content" />);
    const buildButton = screen.getByTestId("plan-preview-build-button");

    // Act
    await user.click(buildButton);

    // Assert
    expect(useConversationStore.getState().conversationMode).toBe("code");
  });

  it("should send build prompt message when Build button is clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const expectedPrompt =
      "Execute the plan based on the workspace/project/PLAN.md file.";
    renderWithProviders(<PlanPreview planContent="Plan content" />);
    const buildButton = screen.getByTestId("plan-preview-build-button");

    // Act
    await user.click(buildButton);

    // Assert
    expect(createChatMessage).toHaveBeenCalledTimes(1);
    expect(createChatMessage).toHaveBeenCalledWith(
      expectedPrompt,
      [],
      [],
      expect.any(String),
    );
    expect(mockSend).toHaveBeenCalledTimes(1);
    expect(mockSend).toHaveBeenCalledWith(
      expect.objectContaining({
        action: "message",
        args: expect.objectContaining({
          content: expectedPrompt,
        }),
      }),
    );
  });

  it("should set optimistic user message when Build button is clicked", async () => {
    // Arrange
    useOptimisticUserMessageStore.setState({ optimisticUserMessage: null });
    const user = userEvent.setup();
    const expectedPrompt =
      "Execute the plan based on the workspace/project/PLAN.md file.";
    renderWithProviders(<PlanPreview planContent="Plan content" />);
    const buildButton = screen.getByTestId("plan-preview-build-button");

    // Act
    await user.click(buildButton);

    // Assert
    expect(useOptimisticUserMessageStore.getState().optimisticUserMessage).toBe(
      expectedPrompt,
    );
  });

  it("should render header with PLAN_MD text", () => {
    const { container } = renderWithProviders(
      <PlanPreview planContent="Plan content" />,
    );

    // Check that the translation key is rendered (i18n mock returns the key)
    expect(container.textContent).toContain("COMMON$PLAN_MD");
  });

  it("should render plan content", () => {
    const planContent = `# Heading 1
## Heading 2
- List item 1
- List item 2

**Bold text** and *italic text*`;

    const { container } = renderWithProviders(
      <PlanPreview planContent={planContent} />,
    );

    expect(container.textContent).toContain("Heading 1");
    expect(container.textContent).toContain("Heading 2");
  });
});
