import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ChatInterfaceWrapper } from "#/components/features/conversation/conversation-main/chat-interface-wrapper";

vi.mock("#/components/features/chat/chat-interface", () => ({
  ChatInterface: () => <div data-testid="chat-interface" />,
}));

describe("ChatInterfaceWrapper", () => {
  it("renders the chat interface when the right panel is hidden", () => {
    render(<ChatInterfaceWrapper isRightPanelShown={false} />);

    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("renders the chat interface when the right panel is shown", () => {
    render(<ChatInterfaceWrapper isRightPanelShown />);

    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });
});
