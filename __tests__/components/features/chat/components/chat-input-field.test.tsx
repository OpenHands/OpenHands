import React from "react";
import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ChatInputField } from "#/components/features/chat/components/chat-input-field";

function Harness({
  disabled,
  placeholder,
}: {
  disabled: boolean;
  placeholder?: string;
}) {
  const ref = React.useRef<HTMLDivElement | null>(null);
  return (
    <ChatInputField
      chatInputRef={ref}
      disabled={disabled}
      placeholder={placeholder}
      onInput={vi.fn()}
      onPaste={vi.fn()}
      onKeyDown={vi.fn()}
    />
  );
}

describe("ChatInputField auto-focus", () => {
  it("focuses the chat input on mount when enabled", () => {
    renderWithProviders(<Harness disabled={false} />);

    expect(screen.getByTestId("chat-input")).toBe(document.activeElement);
  });

  it("does not focus the chat input on mount when disabled", () => {
    renderWithProviders(<Harness disabled />);

    expect(screen.getByTestId("chat-input")).not.toBe(document.activeElement);
  });

  it("does not steal focus when disabled flips from true to false", () => {
    const { rerender } = renderWithProviders(<Harness disabled />);
    document.body.focus();

    rerender(<Harness disabled={false} />);

    expect(screen.getByTestId("chat-input")).not.toBe(document.activeElement);
  });
});

function describeEngineeringTask() {
  return "Describe an engineering task…";
}

describe("ChatInputField placeholder", () => {
  it("falls back to the default build prompt when no placeholder is given", () => {
    renderWithProviders(<Harness disabled={false} />);

    expect(screen.getByTestId("chat-input")).toHaveAttribute(
      "data-placeholder",
      "SUGGESTIONS$WHAT_TO_BUILD",
    );
  });

  it("uses the caller-supplied placeholder when provided", () => {
    renderWithProviders(
      <Harness disabled={false} placeholder={describeEngineeringTask()} />,
    );

    expect(screen.getByTestId("chat-input")).toHaveAttribute(
      "data-placeholder",
      describeEngineeringTask(),
    );
  });
});
