import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CompactContextButton } from "./compact-context-button";
import { ContextMeter } from "./context-meter";

vi.mock("#/hooks/use-compact-context-action", () => ({
  useCompactContextAction: () => ({
    handleCompact: vi.fn(),
    isCompacting: false,
    isDisabled: false,
    description: "Compact context",
  }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

// The percentage math itself (zero-window guard, 100% cap) is covered by
// __tests__/utils/format-token-count.test.ts; these tests cover rendering.
describe("ContextMeter", () => {
  it("renders the raw token numbers", () => {
    render(<ContextMeter perTurnToken={17994} contextWindow={1000000} />);

    expect(screen.getByTestId("context-meter")).toHaveTextContent(
      `${(17994).toLocaleString()} / ${(1000000).toLocaleString()}`,
    );
  });

  it("stays neutral below the warning threshold", () => {
    render(<ContextMeter perTurnToken={50} contextWindow={100} />);

    const bar = screen.getByTestId("context-meter-bar");
    expect(bar).toHaveClass("bg-foreground");
    expect(bar).toHaveClass("rounded-full");
    expect(bar.style.width).toBe("50%");
  });

  it("warns above 70% fill", () => {
    render(<ContextMeter perTurnToken={75} contextWindow={100} />);

    expect(screen.getByTestId("context-meter-bar")).toHaveClass(
      "bg-feedback-warning",
    );
    expect(screen.getByText(/75%/)).toHaveClass("text-feedback-warning");
  });

  it("signals danger above 90% fill", () => {
    render(<ContextMeter perTurnToken={95} contextWindow={100} />);

    expect(screen.getByTestId("context-meter-bar")).toHaveClass(
      "bg-status-error",
    );
    expect(screen.getByText(/95%/)).toHaveClass("text-feedback-error");
    expect(screen.getByTestId("context-meter")).toHaveTextContent(
      "95% CONVERSATION$USED (5% CONVERSATION$LEFT)",
    );
  });

  it("caps the bar width at 100%", () => {
    render(<ContextMeter perTurnToken={150} contextWindow={100} />);

    expect(screen.getByTestId("context-meter-bar").style.width).toBe("100%");
  });

  it("shows an unknown-window state instead of 'x / 0' when the window is unreported", () => {
    render(<ContextMeter perTurnToken={31778} contextWindow={0} />);

    const meter = screen.getByTestId("context-meter");
    expect(meter).toHaveTextContent("CONVERSATION$CONTEXT_WINDOW_UNKNOWN");
    expect(meter).toHaveTextContent((31778).toLocaleString());
    expect(meter).not.toHaveTextContent("/ 0");
    expect(meter).not.toHaveTextContent("0.0%");
  });
});

describe("CompactContextButton", () => {
  it("uses warning feedback color when context is above the threshold", () => {
    render(<CompactContextButton fillPercent={75} />);

    expect(screen.getByText("CONVERSATION$CONTEXT_FILLING_UP")).toHaveClass(
      "text-feedback-warning",
    );
  });
});
