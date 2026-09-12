/**
 * Component tests for the new uploading state in ChatMessage (#16430).
 * Run: `npx vitest run __tests__/components/chat-message-uploading`
 */
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ChatMessage } from "#/components/features/chat/chat-message";

describe("ChatMessage — uploading state (#16430)", () => {
  it("renders uploading shimmer with no progress when uploadProgress is 0", () => {
    render(
      <ChatMessage type="user" message="hello" pendingStatus="uploading" uploadProgress={0} />,
    );
    expect(screen.getByTestId("chat-message-uploading")).toBeInTheDocument();
    // Progress bar exists
    const bar = screen.getByTestId("chat-message-upload-progress-bar");
    expect(bar).toHaveStyle({ width: "0%" });
  });

  it("renders uploading shimmer with correct progress width", () => {
    render(
      <ChatMessage type="user" message="attach" pendingStatus="uploading" uploadProgress={65} />,
    );
    const bar = screen.getByTestId("chat-message-upload-progress-bar");
    expect(bar).toHaveStyle({ width: "65%" });
  });

  it("renders uploading state with accessible progressbar role", () => {
    render(
      <ChatMessage type="user" message="file" pendingStatus="uploading" uploadProgress={30} />,
    );
    const progressbar = screen.getByRole("progressbar");
    expect(progressbar).toHaveAttribute("aria-valuenow", "30");
    expect(progressbar).toHaveAttribute("aria-valuemin", "0");
    expect(progressbar).toHaveAttribute("aria-valuemax", "100");
  });

  it("renders the user message text inside the bubble while uploading", () => {
    render(
      <ChatMessage type="user" message="my prompt" pendingStatus="uploading" uploadProgress={10} />,
    );
    expect(screen.getByText("my prompt")).toBeInTheDocument();
  });

  it("does NOT render chat-message-uploading when status is sending", () => {
    render(<ChatMessage type="user" message="hello" pendingStatus="sending" />);
    expect(screen.queryByTestId("chat-message-uploading")).not.toBeInTheDocument();
    expect(screen.getByTestId("chat-message-sending")).toBeInTheDocument();
  });

  it("does NOT render uploading section for non-user messages", () => {
    render(
      <ChatMessage type="agent" message="agent reply" pendingStatus="uploading" uploadProgress={50} />,
    );
    // uploading branch only fires for type==="user"
    expect(screen.queryByTestId("chat-message-uploading")).not.toBeInTheDocument();
  });

  it("applies opacity-60 class to the bubble article in uploading state", () => {
    render(
      <ChatMessage type="user" message="dim me" pendingStatus="uploading" uploadProgress={0} />,
    );
    const bubble = screen.getByTestId("user-message");
    expect(bubble.className).toContain("opacity-60");
  });
});
