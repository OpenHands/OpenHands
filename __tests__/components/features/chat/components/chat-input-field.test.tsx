import React from "react";
import { screen, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ChatInputField } from "#/components/features/chat/components/chat-input-field";
import { ContextWindowRing } from "#/components/features/chat/components/context-window-ring";
import { COLOR_THEMES } from "#/themes/color-themes";
import { AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES } from "#/styles/agent-server-ui-style-scope";
import {
  hasConversationStarted,
  resolvePickerKind,
} from "#/components/features/chat/components/resolve-picker-kind";

describe("chat-input-field", () => {
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

    it("uses the provided placeholder when one is passed", () => {
      renderWithProviders(
        <Harness disabled={false} placeholder="Custom placeholder" />,
      );

      expect(screen.getByTestId("chat-input")).toHaveAttribute(
        "data-placeholder",
        "Custom placeholder",
      );
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
});

describe("context-window-ring", () => {
  /**
   * WCAG 2.1 SC 1.4.11 asks 3:1 for non-text contrast. The ring's track carries
   * information (the arc's proportion is only readable against it), so it is held
   * to that bar rather than treated as chrome.
   */
  const MIN_RATIO = 3;

  /**
   * Themes are read from `color-themes.ts`, not from a stylesheet. `index.css`
   * ships the deepsea scale but `DEFAULT_COLOR_THEME` overrides it at runtime, so
   * a stylesheet-derived assertion validates a palette no default install renders.
   */
  const SURFACE_STOP = "--cool-grey-925"; // --oh-surface, the composer background
  const FOREGROUND_STOP = "--cool-grey-100"; // --oh-foreground, the neutral arc

  const HOVER_ALPHA = 0.1;

  function channels(hex: string): [number, number, number] {
    const h = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [
      number,
      number,
      number,
    ];
  }

  function relativeLuminance(hex: string): number {
    const [r, g, b] = channels(hex).map((channel) => {
      const c = channel / 255;
      return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  function contrastRatio(a: string, b: string): number {
    const [high, low] = [relativeLuminance(a), relativeLuminance(b)].sort(
      (x, y) => y - x,
    );
    return (high + 0.05) / (low + 0.05);
  }

  /** Composite a translucent overlay onto an opaque backdrop. */
  function composite(overlay: string, backdrop: string, alpha: number): string {
    const o = channels(overlay);
    const b = channels(backdrop);
    return `#${o
      .map((channel, i) =>
        Math.round(alpha * channel + (1 - alpha) * b[i])
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")}`.toUpperCase();
  }

  describe("ContextWindowRing", () => {
    it("draws the track and the arc in different tones", () => {
      render(<ContextWindowRing percentage={5} />);

      const track = screen.getByTestId("context-window-ring-track");
      const arc = screen.getByTestId("context-window-ring-arc");

      expect(track.style.stroke).not.toEqual(arc.getAttribute("stroke"));
    });

    it("derives the track from the foreground rather than a scale stop", () => {
      render(<ContextWindowRing percentage={5} />);

      // A stop-pinned track cannot hold across palettes; see the constant's docs.
      expect(
        screen.getByTestId("context-window-ring-track").style.stroke,
      ).toEqual(expect.stringContaining("var(--oh-context-window-foreground)"));
    });

    /**
     * Read the alpha back out of what the component actually renders, so the
     * contrast cases below describe the shipped track rather than a constant that
     * could drift away from it.
     */
    function renderedTrackColor(): string {
      render(<ContextWindowRing percentage={5} />);
      return screen.getByTestId("context-window-ring-track").style.stroke;
    }

    it("renders the alpha it documents", () => {
      expect(renderedTrackColor()).toContain(
        "var(--oh-context-window-track-weight)",
      );
    });

    describe.each(Object.entries(COLOR_THEMES))(
      "contrast under the %s palette",
      (_key, theme) => {
        const surface =
          theme.tokens?.["--oh-surface"] ?? theme.scale[SURFACE_STOP];
        const arc =
          theme.tokens?.["--oh-context-window-foreground"] ??
          theme.scale[FOREGROUND_STOP];
        const alpha =
          Number.parseFloat(
            theme.tokens?.["--oh-context-window-track-weight"] ??
              AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES[
                "--oh-context-window-track-weight"
              ],
          ) / 100;
        const hoverOverlay = theme.tokens?.["--oh-contrast"] ?? "#FFFFFF";
        const hoverFill = composite(hoverOverlay, surface, HOVER_ALPHA);

        it("keeps the track legible against the composer surface", () => {
          const track = composite(arc, surface, alpha);

          expect(contrastRatio(track, surface)).toBeGreaterThanOrEqual(
            MIN_RATIO,
          );
        });

        it("keeps the track legible under the trigger's hover fill", () => {
          const track = composite(arc, hoverFill, alpha);

          expect(contrastRatio(track, hoverFill)).toBeGreaterThanOrEqual(
            MIN_RATIO,
          );
        });

        it("keeps a neutral arc distinguishable from the track in both states", () => {
          expect(
            contrastRatio(arc, composite(arc, surface, alpha)),
          ).toBeGreaterThanOrEqual(MIN_RATIO);
          expect(
            contrastRatio(arc, composite(arc, hoverFill, alpha)),
          ).toBeGreaterThanOrEqual(MIN_RATIO);
        });
      },
    );
  });
});

describe("resolve-picker-kind", () => {
  // The pill is always an LLM selector (OSS-5735): agent-profile switching lives
  // in the "+" tools menu, and neither the backend nor the conversation's state
  // branches here any more — whether a pick is actionable is decided in
  // useChatInputLlmProfileState, so cloud never falls back to a raw model chip.
  describe("resolvePickerKind", () => {
    it("shows the constrained model picker in an ACP context", () => {
      expect(resolvePickerKind({ isAcp: true })).toBe("model");
    });

    it("shows the LLM-profile picker for an OpenHands context", () => {
      expect(resolvePickerKind({ isAcp: false })).toBe("llm-profile");
    });
  });

  describe("hasConversationStarted", () => {
    const empty = {
      isLoadingHistory: false,
      hasUserEvents: false,
      hasPendingUserMessages: false,
      hasSubstantiveAgentActions: false,
      hasModelEntries: false,
    };

    it("keeps a fully loaded blank conversation in pre-run state", () => {
      expect(hasConversationStarted(empty)).toBe(false);
    });

    it.each([
      "isLoadingHistory",
      "hasUserEvents",
      "hasPendingUserMessages",
      "hasSubstantiveAgentActions",
      "hasModelEntries",
    ] as const)("treats %s as started", (field) => {
      expect(hasConversationStarted({ ...empty, [field]: true })).toBe(true);
    });
  });
});
