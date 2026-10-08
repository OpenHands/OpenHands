import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import React from "react";
import { ActionTooltip } from "#/components/shared/action-tooltip";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      if (key === "CHAT_INTERFACE$INPUT_CONTINUE_MESSAGE") return "Continue";
      if (key === "BUTTON$CANCEL") return "Cancel";
      return key;
    },
  }),
}));

describe("ActionTooltip", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders Mac shortcut hints on Apple platforms", () => {
    vi.spyOn(navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    );

    const { rerender } = render(
      <ActionTooltip type="confirm" onClick={vi.fn()} />,
    );
    expect(screen.getByTestId("action-confirm-button")).toHaveTextContent("⌘");

    rerender(<ActionTooltip type="reject" onClick={vi.fn()} />);
    expect(screen.getByTestId("action-reject-button")).toHaveTextContent("⌘");
  });

  it("renders Ctrl shortcut hints on non-Apple platforms", () => {
    vi.spyOn(navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (X11; Linux x86_64)",
    );

    const { rerender } = render(
      <ActionTooltip type="confirm" onClick={vi.fn()} />,
    );
    expect(screen.getByTestId("action-confirm-button")).toHaveTextContent(
      "Ctrl",
    );

    rerender(<ActionTooltip type="reject" onClick={vi.fn()} />);
    expect(screen.getByTestId("action-reject-button")).toHaveTextContent(
      "Ctrl",
    );
  });
});
