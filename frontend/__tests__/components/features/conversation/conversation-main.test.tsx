import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mutable mock state for controlling viewport width
let mockWidth = 1200;

// Track ChatInterface unmount via vi.fn()
const chatInterfaceUnmount = vi.fn();

vi.mock("@uidotdev/usehooks", () => ({
  useWindowSize: () => ({ width: mockWidth, height: 800 }),
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
    mockWidth = 1200;
    chatInterfaceUnmount.mockClear();
  });

  it("renders ChatInterface at desktop width", () => {
    mockWidth = 1200;
    render(<ConversationMain />);
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("renders ChatInterface at mobile width", () => {
    mockWidth = 800;
    render(<ConversationMain />);
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("does not unmount ChatInterface when crossing from desktop to mobile", () => {
    mockWidth = 1200;
    const { rerender } = render(<ConversationMain />);
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();

    // Cross the 1024px breakpoint to mobile
    mockWidth = 800;
    rerender(<ConversationMain />);

    // ChatInterface must NOT have been unmounted and remounted
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("does not unmount ChatInterface when crossing from mobile to desktop", () => {
    mockWidth = 800;
    const { rerender } = render(<ConversationMain />);
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();

    // Cross the 1024px breakpoint to desktop
    mockWidth = 1200;
    rerender(<ConversationMain />);

    // ChatInterface must NOT have been unmounted and remounted
    expect(chatInterfaceUnmount).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("survives rapid back-and-forth resize without unmounting ChatInterface", () => {
    mockWidth = 1200;
    const { rerender } = render(<ConversationMain />);

    // Simulate rapid resize back and forth across the breakpoint
    for (const width of [800, 1200, 800, 1200, 800]) {
      mockWidth = width;
      rerender(<ConversationMain />);
    }

    expect(chatInterfaceUnmount).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });
});
