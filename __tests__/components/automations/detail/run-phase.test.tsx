import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";

import {
  resolveRunPhaseText,
  RunPhase,
} from "#/components/features/automations/detail/run-phase";

describe("RunPhase — current_phase", () => {
  it("shows the user-facing string as-is", () => {
    render(<RunPhase currentPhase="Examining the diff" />);

    expect(screen.getByTestId("run-phase")).toHaveTextContent(
      "Examining the diff",
    );
  });

  it("shows emoji and non-Latin text as-is", () => {
    render(<RunPhase currentPhase="🔍 Опрашиваем PR-ы" />);

    expect(screen.getByText("🔍 Опрашиваем PR-ы")).toBeInTheDocument();
  });

  it.each([
    ["undefined (an older service)", undefined],
    ["null", null],
    ["empty string", ""],
    ["whitespace only", "   "],
  ])(
    "renders nothing and never touches the console with %s",
    (_case, phase) => {
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      expect(() => render(<RunPhase currentPhase={phase} />)).not.toThrow();

      expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
      expect(errorSpy).not.toHaveBeenCalled();
      expect(warnSpy).not.toHaveBeenCalled();
    },
  );

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("boundary: a 200-character phase (the contract's max) reaches the DOM whole, so truncation stays visual", () => {
    const phase = "x".repeat(200);

    render(<RunPhase currentPhase={phase} />);

    expect(screen.getByTestId("run-phase")).toHaveTextContent(phase);
  });
});

describe("resolveRunPhaseText — one answer for the row and its tooltip", () => {
  it("resolves current_phase as-is", () => {
    expect(resolveRunPhaseText("Preparing environment")).toBe(
      "Preparing environment",
    );
  });

  it("trims padding", () => {
    expect(resolveRunPhaseText("  Queued  ")).toBe("Queued");
  });

  it("resolves to null when there is nothing to show", () => {
    expect(resolveRunPhaseText(undefined)).toBeNull();
    expect(resolveRunPhaseText(null)).toBeNull();
    expect(resolveRunPhaseText("")).toBeNull();
    expect(resolveRunPhaseText("   ")).toBeNull();
  });
});

describe("RunPhase — reachable without a mouse", () => {
  it("leaves the whole phase in the accessibility tree", () => {
    const phase = "x".repeat(200);

    render(<RunPhase currentPhase={phase} />);

    const text = screen.getByTestId("run-phase");
    expect(text).toHaveTextContent(phase);
    expect(text).not.toHaveAttribute("aria-hidden");
    expect(text.closest("[tabindex]")).toBeNull();
  });
});
