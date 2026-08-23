import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ConversationOverviewToggle } from "#/components/features/conversation/conversation-overview-toggle";
import { useConversationStore } from "#/stores/conversation-store";

const { breakpointIsMobile } = vi.hoisted(() => ({
  breakpointIsMobile: { value: false },
}));

vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: () => breakpointIsMobile.value,
}));

vi.mock("#/hooks/use-is-archived-conversation", () => ({
  useIsArchivedConversation: () => false,
}));

vi.mock("#/components/features/conversation/conversation-overview-panel", () => ({
  ConversationOverviewPanel: () => (
    <div data-testid="conversation-overview-panel" />
  ),
}));

describe("ConversationOverviewToggle", () => {
  beforeEach(() => {
    breakpointIsMobile.value = false;
    useConversationStore.setState({
      isOverviewPanelShown: false,
      isOverviewPanelPeeked: false,
      isRightPanelShown: false,
      hasRightPanelToggled: false,
    });
  });

  it("toggles the overview panel when clicked", async () => {
    const user = userEvent.setup();
    render(<ConversationOverviewToggle />);

    await user.click(screen.getByTestId("conversation-overview-toggle"));
    expect(useConversationStore.getState().isOverviewPanelShown).toBe(true);

    await user.click(screen.getByTestId("conversation-overview-toggle"));
    expect(useConversationStore.getState().isOverviewPanelShown).toBe(false);
  });

  it("closes the files drawer and shows overview when the files drawer is open", async () => {
    const user = userEvent.setup();
    useConversationStore.setState({
      isOverviewPanelShown: false,
      isRightPanelShown: true,
      hasRightPanelToggled: true,
    });
    render(<ConversationOverviewToggle />);

    await user.click(screen.getByTestId("conversation-overview-toggle"));

    const state = useConversationStore.getState();
    expect(state.isRightPanelShown).toBe(false);
    expect(state.hasRightPanelToggled).toBe(false);
    expect(state.isOverviewPanelShown).toBe(true);
  });

  it("closes the overview panel when the right drawer opens", () => {
    useConversationStore.setState({
      isOverviewPanelShown: true,
      isRightPanelShown: false,
    });

    const { rerender } = render(<ConversationOverviewToggle />);
    expect(useConversationStore.getState().isOverviewPanelShown).toBe(true);

    useConversationStore.setState({ isRightPanelShown: true });
    rerender(<ConversationOverviewToggle />);

    expect(useConversationStore.getState().isOverviewPanelShown).toBe(false);
  });

  it("peeks the overview on hover while the right drawer is open", async () => {
    const user = userEvent.setup();
    useConversationStore.setState({
      isOverviewPanelShown: false,
      isRightPanelShown: true,
    });
    render(<ConversationOverviewToggle />);

    await user.hover(screen.getByTestId("conversation-overview-toggle"));

    expect(useConversationStore.getState().isOverviewPanelPeeked).toBe(true);
    expect(useConversationStore.getState().isOverviewPanelShown).toBe(false);
    expect(screen.getByTestId("conversation-overview-peek")).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-panel"),
    ).toBeInTheDocument();
  });

  it("does not peek the overview on hover when the right drawer is closed", async () => {
    const user = userEvent.setup();
    render(<ConversationOverviewToggle />);

    await user.hover(screen.getByTestId("conversation-overview-toggle"));

    expect(useConversationStore.getState().isOverviewPanelPeeked).toBe(false);
    expect(
      screen.queryByTestId("conversation-overview-peek"),
    ).not.toBeInTheDocument();
  });

  it("stays visible and supports hover peek on smaller screens", async () => {
    const user = userEvent.setup();
    breakpointIsMobile.value = true;
    useConversationStore.setState({
      isOverviewPanelShown: false,
      isRightPanelShown: true,
    });
    render(<ConversationOverviewToggle />);

    const toggle = screen.getByTestId("conversation-overview-toggle");
    expect(toggle).toBeInTheDocument();

    await user.hover(toggle);

    expect(useConversationStore.getState().isOverviewPanelPeeked).toBe(true);
    expect(screen.getByTestId("conversation-overview-peek")).toBeInTheDocument();
  });
});
