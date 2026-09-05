import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ChatInterfaceWrapper } from "#/components/features/conversation/conversation-main/chat-interface-wrapper";
import { useConversationStore } from "#/stores/conversation-store";
import type { ConversationOverviewLayoutMode } from "#/components/features/conversation/conversation-overview-panel.constants";
import { CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX } from "#/components/features/conversation/conversation-overview-panel.constants";

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

const mockUseConversationOverviewLayoutMode = vi.fn(
  (): ConversationOverviewLayoutMode => "inline",
);

vi.mock("#/hooks/use-conversation-overview-layout-mode", () => ({
  useConversationOverviewLayoutMode: () =>
    mockUseConversationOverviewLayoutMode(),
}));

describe("ChatInterfaceWrapper", () => {
  beforeEach(() => {
    mockUseConversationOverviewLayoutMode.mockReturnValue("inline");
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

  it("uses the inline overview layout when space is available", () => {
    useConversationStore.setState({ isOverviewPanelShown: true });
    render(<ChatInterfaceWrapper isRightPanelShown={false} />);

    const column = screen.getByTestId("conversation-overview-column");
    expect(column).toHaveAttribute("data-layout-mode", "inline");
    expect(column).toHaveClass("absolute");
    expect(screen.getByTestId("conversation-thread-column")).toHaveStyle({
      paddingRight: `${CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX}px`,
    });
    expect(screen.getByTestId("conversation-overview-panel")).toBeInTheDocument();
  });

  it("overlays overview in the right margin on wide layouts", () => {
    mockUseConversationOverviewLayoutMode.mockReturnValue("overlay");
    useConversationStore.setState({ isOverviewPanelShown: true });

    render(<ChatInterfaceWrapper isRightPanelShown={false} />);

    const column = screen.getByTestId("conversation-overview-column");
    expect(column).toHaveAttribute("data-layout-mode", "overlay");
    expect(column).toHaveClass("absolute");
    expect(screen.getByTestId("conversation-thread-column")).toHaveStyle({
      paddingRight: "0px",
    });
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
    mockUseConversationOverviewLayoutMode.mockReturnValue("hidden");
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
