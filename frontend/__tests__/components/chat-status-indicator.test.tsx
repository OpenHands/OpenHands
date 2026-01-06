import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import ChatStatusIndicator from "#/components/features/chat/chat-status-indicator";

vi.mock("#/icons/debug-stackframe-dot.svg?react", () => ({
  default: (props: any) => (
    <svg data-testid="debug-stackframe-dot" {...props} />
  ),
}));

describe("ChatStatusIndicator", () => {
  it("renders the status indicator with status text", () => {
    render(
      <ChatStatusIndicator
        status="Waiting for sandbox"
        statusColor="#FFD600"
      />
    );

    expect(
      screen.getByTestId("chat-status-indicator"),
    ).toBeInTheDocument();
    expect(screen.getByText("Waiting for sandbox")).toBeInTheDocument();
  });

  it("passes the statusColor to the DebugStackframeDot icon", () => {
    render(
      <ChatStatusIndicator
        status="Error"
        statusColor="#FF684E"
      />
    );

    const icon = screen.getByTestId("debug-stackframe-dot");
    expect(icon).toHaveAttribute("color", "#FF684E");
  });

  it("renders the DebugStackframeDot icon", () => {
    render(
      <ChatStatusIndicator
        status="Loading"
        statusColor="#FFD600"
      />
    );

    expect(screen.getByTestId("debug-stackframe-dot")).toBeInTheDocument();
  });
});
