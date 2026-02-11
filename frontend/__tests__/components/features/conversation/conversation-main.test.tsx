import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mutable mock state for controlling breakpoint
let mockIsMobile = false;

// Track ChatInterface unmount via vi.fn()
const chatInterfaceUnmount = vi.fn();

vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: () => mockIsMobile,
}));

vi.mock("#/hooks/use-resizable-panels", () => ({
  useResizablePanels: () => ({
    leftWidth: 50,
    rightWidth: 50,
    isDragging: false,
    containerRef: { current: null },
    handleMouseDown: vi.fn(),
  }),
}));

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: () => ({
    isRightPanelShown: false,
  }),
}));

// Mock ChatInterface with useEffect to track mount/unmount lifecycle
vi.mock("#/components/features/chat/chat-interface", () => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const React = require("react");
  return {
    ChatInterface: () => {
      React.useEffect(() => {
        return () => chatInterfaceUnmount();
      }, []);
      return <div data-testid="chat-interface">Chat Interface</div>;
    },
  };
});

vi.mock(
  "#/components/features/conversation/conversation-tabs/conversation-tab-content/conversation-tab-content",
  () => ({
    ConversationTabContent: () => <div data-testid="tab-content" />,
  }),
);

import { ConversationMain } from "#/components/features/conversation/conversation-main/conversation-main";

describe("ConversationMain - Layout Transition Stability", () => {
  beforeEach(() => {
    mockIsMobile = false;
    chatInterfaceUnmount.mockClear();
  });

  it("renders ChatInterface at desktop width", () => {
    mockIsMobile = false;
    render(<ConversationMain />);
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("renders ChatInterface at mobile width", () => {
    mockIsMobile = true;
    render(<ConversationMain />);
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("does not unmount ChatInterface when crossing from desktop to mobile", () => {
    mockIsMobile = false;
    const { rerender } = render(<ConversationMain />);
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();

    // Cross the breakpoint to mobile
    mockIsMobile = true;
    rerender(<ConversationMain />);

    // ChatInterface must NOT have been unmounted and remounted
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("does not unmount ChatInterface when crossing from mobile to desktop", () => {
    mockIsMobile = true;
    const { rerender } = render(<ConversationMain />);
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();

    // Cross the breakpoint to desktop
    mockIsMobile = false;
    rerender(<ConversationMain />);

    // ChatInterface must NOT have been unmounted and remounted
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("survives rapid back-and-forth resize without unmounting ChatInterface", () => {
    mockIsMobile = false;
    const { rerender } = render(<ConversationMain />);

    // Simulate rapid resize back and forth across the breakpoint
    for (const mobile of [true, false, true, false, true]) {
      mockIsMobile = mobile;
      rerender(<ConversationMain />);
    }

    expect(chatInterfaceUnmount).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });
});
