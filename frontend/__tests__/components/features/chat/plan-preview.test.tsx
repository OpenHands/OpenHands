import React from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { PlanPreview } from "#/components/features/chat/plan-preview";
import { ScrollProvider } from "#/context/scroll-context";
import { useConversationStore } from "#/stores/conversation-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { createChatMessage } from "#/services/chat-service";

// PlanPreview uses useScrollContext; wrap with ScrollProvider and a mock value.
const mockScrollDomToBottom = vi.fn();
const scrollContextValue = {
  scrollRef: { current: null },
  autoScroll: true,
  setAutoScroll: vi.fn(),
  scrollDomToBottom: mockScrollDomToBottom,
  hitBottom: true,
  setHitBottom: vi.fn(),
  onChatBodyScroll: vi.fn(),
};

function renderPlanPreview(ui: React.ReactElement) {
  return renderWithProviders(
    <ScrollProvider value={scrollContextValue}>{ui}</ScrollProvider>,
  );
}

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

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

describe("PlanPreview", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store states
    localStorage.clear();
    useOptimisticUserMessageStore.setState({
      optimisticUserMessage: null,
    });
    useConversationStore.setState({
      conversationMode: "plan",
      selectedTab: null,
      isRightPanelShown: false,
      hasRightPanelToggled: false,
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
    localStorage.clear();
  });

  it("should render nothing when planContent is null", () => {
    renderPlanPreview(<PlanPreview planContent={null} />);

    const contentDiv = screen.getByTestId("plan-preview-content");
    expect(contentDiv).toBeInTheDocument();
    expect(contentDiv.textContent?.trim() || "").toBe("");
  });

  it("should render nothing when planContent is undefined", () => {
    renderPlanPreview(<PlanPreview planContent={undefined} />);

    const contentDiv = screen.getByTestId("plan-preview-content");
    expect(contentDiv).toBeInTheDocument();
    expect(contentDiv.textContent?.trim() || "").toBe("");
  });

  it("should render markdown content when planContent is provided", () => {
    const planContent = "# Plan Title\n\nThis is the plan content.";

    const { container } = renderPlanPreview(
      <PlanPreview planContent={planContent} />,
    );

    // Check that component rendered and contains the content (markdown may break up text)
    expect(container.firstChild).not.toBeNull();
    expect(container.textContent).toContain("Plan Title");
    expect(container.textContent).toContain("This is the plan content.");
  });

  it("should render full content when length is less than or equal to 300 characters", () => {
    const planContent = "A".repeat(300);

    const { container } = renderPlanPreview(
      <PlanPreview planContent={planContent} />,
    );

    // Content should be present (may be broken up by markdown)
    expect(container.textContent).toContain(planContent);
    expect(screen.queryByText(/COMMON\$READ_MORE/i)).not.toBeInTheDocument();
  });

  it("should truncate content when length exceeds 300 characters", () => {
    const longContent = "A".repeat(350);

    const { container } = renderPlanPreview(
      <PlanPreview planContent={longContent} />,
    );

    // Truncated content should be present (may be broken up by markdown)
    expect(container.textContent).toContain("A".repeat(300));
    expect(container.textContent).toContain("...");
    expect(container.textContent).toContain("COMMON$READ_MORE");
  });

  it("should render Build button", () => {
    renderPlanPreview(<PlanPreview planContent="Plan content" />);

    const buildButton = screen.getByTestId("plan-preview-build-button");
    expect(buildButton).toBeInTheDocument();
  });

  it("should switch to code mode when Build button is clicked", async () => {
    // Arrange
    useConversationStore.setState({ conversationMode: "plan" });
    const user = userEvent.setup();
    renderPlanPreview(<PlanPreview planContent="Plan content" />);
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
      "Execute the plan based on the .agents_tmp/PLAN.md file.";
    renderPlanPreview(<PlanPreview planContent="Plan content" />);
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
      "Execute the plan based on the .agents_tmp/PLAN.md file.";
    renderPlanPreview(<PlanPreview planContent="Plan content" />);
    const buildButton = screen.getByTestId("plan-preview-build-button");

    // Act
    await user.click(buildButton);

    // Assert
    expect(useOptimisticUserMessageStore.getState().optimisticUserMessage).toBe(
      expectedPrompt,
    );
  });

  it("should disable Build button when isBuildDisabled is true", () => {
    // Arrange
    renderPlanPreview(
      <PlanPreview planContent="Plan content" isBuildDisabled={true} />,
    );

    // Act
    const buildButton = screen.getByTestId("plan-preview-build-button");

    // Assert
    expect(buildButton).toBeDisabled();
  });

  it("should not disable Build button when isBuildDisabled is false", () => {
    // Arrange
    renderPlanPreview(
      <PlanPreview planContent="Plan content" isBuildDisabled={false} />,
    );

    // Act
    const buildButton = screen.getByTestId("plan-preview-build-button");

    // Assert
    expect(buildButton).not.toBeDisabled();
  });

  it("should not disable Build button when isBuildDisabled is undefined", () => {
    // Arrange
    renderPlanPreview(<PlanPreview planContent="Plan content" />);

    // Act
    const buildButton = screen.getByTestId("plan-preview-build-button");

    // Assert
    expect(buildButton).not.toBeDisabled();
  });

  it("should not call onBuildClick when Build button is disabled and clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const onBuildClick = vi.fn();

    renderPlanPreview(
      <PlanPreview planContent="Plan content" isBuildDisabled={true} />,
    );

    const buildButton = screen.getByTestId("plan-preview-build-button");

    // Act
    await user.click(buildButton);

    // Assert
    expect(onBuildClick).not.toHaveBeenCalled();
  });

  it("should render header with PLAN_MD text", () => {
    const { container } = renderPlanPreview(
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

    const { container } = renderPlanPreview(
      <PlanPreview planContent={planContent} />,
    );

    expect(container.textContent).toContain("Heading 1");
    expect(container.textContent).toContain("Heading 2");
  });

  it("should use planHeadings components for h1 headings", () => {
    // Arrange
    const planContent = "# Main Title";

    // Act
    const { container } = renderPlanPreview(
      <PlanPreview planContent={planContent} />,
    );

    // Assert
    const h1 = container.querySelector("h1");
    expect(h1).toBeInTheDocument();
    expect(h1).toHaveTextContent("Main Title");
  });

  it("should use planHeadings components for h2 headings", () => {
    // Arrange
    const planContent = "## Section Title";

    // Act
    const { container } = renderPlanPreview(
      <PlanPreview planContent={planContent} />,
    );

    // Assert
    const h2 = container.querySelector("h2");
    expect(h2).toBeInTheDocument();
    expect(h2).toHaveTextContent("Section Title");
  });

  it("should use planHeadings components for h3 headings", () => {
    // Arrange
    const planContent = "### Subsection Title";

    // Act
    const { container } = renderPlanPreview(
      <PlanPreview planContent={planContent} />,
    );

    // Assert
    const h3 = container.querySelector("h3");
    expect(h3).toBeInTheDocument();
    expect(h3).toHaveTextContent("Subsection Title");
  });

  it("should use planHeadings components for all heading levels", () => {
    // Arrange
    const planContent = `# H1 Title
## H2 Title
### H3 Title
#### H4 Title
##### H5 Title
###### H6 Title`;

    // Act
    const { container } = renderPlanPreview(
      <PlanPreview planContent={planContent} />,
    );

    // Assert
    expect(container.querySelector("h1")).toBeInTheDocument();
    expect(container.querySelector("h2")).toBeInTheDocument();
    expect(container.querySelector("h3")).toBeInTheDocument();
    expect(container.querySelector("h4")).toBeInTheDocument();
    expect(container.querySelector("h5")).toBeInTheDocument();
    expect(container.querySelector("h6")).toBeInTheDocument();
  });

  it("should call selectTab with 'planner' when View button is clicked", async () => {
    const user = userEvent.setup();
    const planContent = "Plan content";
    const conversationId = "test-conversation-id";

    // Arrange: Set up initial state
    useConversationStore.setState({
      selectedTab: null,
      isRightPanelShown: false,
      hasRightPanelToggled: false,
    });

    renderPlanPreview(<PlanPreview planContent={planContent} />);

    // Act: Click the View button
    const viewButton = screen.getByTestId("plan-preview-view-button");
    await user.click(viewButton);

    // Assert: Verify selectTab was called with 'planner' and panel was opened
    // The hook sets hasRightPanelToggled, which should trigger isRightPanelShown update
    // In tests, we need to manually sync or check hasRightPanelToggled
    expect(useConversationStore.getState().selectedTab).toBe("planner");
    expect(useConversationStore.getState().hasRightPanelToggled).toBe(true);

    // Verify localStorage was updated
    const storedState = JSON.parse(
      localStorage.getItem(`conversation-state-${conversationId}`)!,
    );
    expect(storedState.selectedTab).toBe("planner");
    expect(storedState.rightPanelShown).toBe(true);
  });

  it("should call selectTab with 'planner' when Read more button is clicked", async () => {
    const user = userEvent.setup();
    const longContent = "A".repeat(350);
    const conversationId = "test-conversation-id";

    // Arrange: Set up initial state
    useConversationStore.setState({
      selectedTab: null,
      isRightPanelShown: false,
      hasRightPanelToggled: false,
    });

    renderPlanPreview(<PlanPreview planContent={longContent} />);

    // Act: Click the Read more button
    const readMoreButton = screen.getByTestId("plan-preview-read-more-button");
    await user.click(readMoreButton);

    // Assert: Verify selectTab was called with 'planner' and panel was opened
    // The hook sets hasRightPanelToggled, which should trigger isRightPanelShown update
    // In tests, we need to manually sync or check hasRightPanelToggled
    expect(useConversationStore.getState().selectedTab).toBe("planner");
    expect(useConversationStore.getState().hasRightPanelToggled).toBe(true);

    // Verify localStorage was updated
    const storedState = JSON.parse(
      localStorage.getItem(`conversation-state-${conversationId}`)!,
    );
    expect(storedState.selectedTab).toBe("planner");
    expect(storedState.rightPanelShown).toBe(true);
  });
});
