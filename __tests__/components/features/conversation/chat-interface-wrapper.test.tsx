import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ChatInterfaceWrapper } from "#/components/features/conversation/conversation-main/chat-interface-wrapper";
import { useConversationStore } from "#/stores/conversation-store";

vi.mock("#/components/features/chat/chat-interface", () => ({
  ChatInterface: () => <div data-testid="chat-interface" />,
}));

vi.mock("#/components/features/conversation/conversation-overview-panel", () => ({
  ConversationOverviewPanel: () => (
    <div data-testid="conversation-overview-panel" />
  ),
}));

vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: () => false,
}));

const mockUseConversationOverviewColumnSpace = vi.fn(() => true);

vi.mock("#/hooks/use-conversation-overview-column-space", () => ({
  useConversationOverviewColumnSpace: () =>
    mockUseConversationOverviewColumnSpace(),
}));

describe("ChatInterfaceWrapper", () => {
  beforeEach(() => {
    mockUseConversationOverviewColumnSpace.mockReturnValue(true);
    useConversationStore.setState({
      isOverviewPanelShown: false,
      isOverviewPanelPeeked: false,
      isRightPanelShown: false,
    });
  });

  it("renders the chat interface when the right panel is hidden", () => {
    render(<ChatInterfaceWrapper isRightPanelShown={false} />);

    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("renders the chat interface when the right panel is shown", () => {
    render(<ChatInterfaceWrapper isRightPanelShown />);

    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("uses the overview grid layout when space is available", () => {
    useConversationStore.setState({ isOverviewPanelShown: true });
    render(<ChatInterfaceWrapper isRightPanelShown={false} />);

    expect(screen.getByTestId("conversation-overview-column")).toBeInTheDocument();
    expect(screen.getByTestId("conversation-overview-panel")).toBeInTheDocument();
  });

  it("keeps the thread in a height-constrained flex column when overview is shown", () => {
    useConversationStore.setState({ isOverviewPanelShown: true });
    const { container } = render(
      <ChatInterfaceWrapper isRightPanelShown={false} />,
    );

    const threadColumn = container.querySelector(".overflow-hidden.flex-1");
    expect(threadColumn).toBeInTheDocument();
    expect(threadColumn).toHaveClass("min-h-0");
  });

  it("falls back to the centered thread layout when the right column is too narrow", () => {
    mockUseConversationOverviewColumnSpace.mockReturnValue(false);
    useConversationStore.setState({ isOverviewPanelShown: true });

    render(<ChatInterfaceWrapper isRightPanelShown={false} />);

    expect(
      screen.queryByTestId("conversation-overview-column"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-overview-panel"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });
});
