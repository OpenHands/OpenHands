import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RightPanelToggle } from "#/components/features/conversation/right-panel-toggle";
import { useConversationStore } from "#/stores/conversation-store";

const CONVERSATION_ID = "conv-abc123";

const { mockNavigate, breakpointIsMobile } = vi.hoisted(() => ({
  mockNavigate: vi.fn(),
  breakpointIsMobile: { value: false },
}));

vi.mock("react-router", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router")>();
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: () => breakpointIsMobile.value,
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: CONVERSATION_ID }),
}));

vi.mock("#/hooks/use-is-archived-conversation", () => ({
  useIsArchivedConversation: () => false,
}));

describe("RightPanelToggle", () => {
  beforeEach(() => {
    localStorage.clear();
    mockNavigate.mockClear();
    breakpointIsMobile.value = false;
    useConversationStore.setState({
      selectedTab: "files",
      isRightPanelShown: true,
      hasRightPanelToggled: true,
    });
  });

  it("should render the toggle button", () => {
    render(<RightPanelToggle />);
    expect(screen.getByTestId("right-panel-toggle")).toBeInTheDocument();
  });

  it("should hide the panel when clicked while panel is open", async () => {
    const user = userEvent.setup();

    render(<RightPanelToggle />);

    const button = screen.getByTestId("right-panel-toggle");
    await user.click(button);

    const storeState = useConversationStore.getState();
    expect(storeState.hasRightPanelToggled).toBe(false);
    expect(storeState.isRightPanelShown).toBe(false);

    const raw = localStorage.getItem(`conversation-state-${CONVERSATION_ID}`);
    if (raw !== null) {
      expect(JSON.parse(raw)).not.toHaveProperty("rightPanelShown");
    }
  });

  it("should show the panel when clicked while panel is hidden", async () => {
    const user = userEvent.setup();

    useConversationStore.setState({
      isRightPanelShown: false,
      hasRightPanelToggled: false,
    });

    render(<RightPanelToggle />);

    const button = screen.getByTestId("right-panel-toggle");
    await user.click(button);

    const storeState = useConversationStore.getState();
    expect(storeState.hasRightPanelToggled).toBe(true);
    expect(storeState.isRightPanelShown).toBe(true);

    const raw = localStorage.getItem(`conversation-state-${CONVERSATION_ID}`);
    if (raw !== null) {
      expect(JSON.parse(raw)).not.toHaveProperty("rightPanelShown");
    }
  });

  it("should have aria-pressed attribute reflecting panel state on desktop", () => {
    const { unmount } = render(<RightPanelToggle />);
    expect(screen.getByTestId("right-panel-toggle")).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    unmount();

    useConversationStore.setState({ isRightPanelShown: false });
    render(<RightPanelToggle />);
    expect(screen.getByTestId("right-panel-toggle")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("navigates to the full-screen panel route on mobile", async () => {
    breakpointIsMobile.value = true;
    const user = userEvent.setup();

    useConversationStore.setState({
      isRightPanelShown: false,
      hasRightPanelToggled: false,
    });

    render(<RightPanelToggle />);

    await user.click(screen.getByTestId("right-panel-toggle"));

    expect(mockNavigate).toHaveBeenCalledWith(
      `/conversations/${CONVERSATION_ID}/panel`,
    );
    const storeState = useConversationStore.getState();
    expect(storeState.hasRightPanelToggled).toBe(true);
    expect(storeState.isRightPanelShown).toBe(true);
  });
});
