import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ContextMeter } from "./context-meter";

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

    expect(screen.getByTestId("context-meter-bar")).toHaveClass("bg-amber-500");
  });

  it("signals danger above 90% fill", () => {
    render(<ContextMeter perTurnToken={95} contextWindow={100} />);

    expect(screen.getByTestId("context-meter-bar")).toHaveClass("bg-red-500");
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
