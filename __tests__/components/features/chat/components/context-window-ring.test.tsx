import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  ContextWindowRing,
  CONTEXT_WINDOW_RING_TRACK_ALPHA,
} from "#/components/features/chat/components/context-window-ring";
import { COLOR_THEMES } from "#/themes/color-themes";

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

/** `chatInputIconButtonClassName` fills the trigger with `hover:bg-white/10`. */
const HOVER_OVERLAY = "#FFFFFF";
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
    expect(screen.getByTestId("context-window-ring-track").style.stroke).toEqual(
      expect.stringContaining("var(--oh-foreground)"),
    );
  });

  /**
   * Read the alpha back out of what the component actually renders, so the
   * contrast cases below describe the shipped track rather than a constant that
   * could drift away from it.
   */
  function renderedTrackAlpha(): number {
    render(<ContextWindowRing percentage={5} />);
    const { stroke } = screen.getByTestId("context-window-ring-track").style;
    const percent = stroke.match(/var\(--oh-foreground\)\s+([\d.]+)%/);
    if (!percent) {
      throw new Error(`track is not a foreground mix: ${stroke}`);
    }
    return Number(percent[1]) / 100;
  }

  it("renders the alpha it documents", () => {
    expect(renderedTrackAlpha()).toBeCloseTo(CONTEXT_WINDOW_RING_TRACK_ALPHA, 5);
  });

  describe.each(Object.entries(COLOR_THEMES))(
    "contrast under the %s palette",
    (_key, theme) => {
      const surface = theme.scale[SURFACE_STOP];
      const arc = theme.scale[FOREGROUND_STOP];
      const hoverFill = composite(HOVER_OVERLAY, surface, HOVER_ALPHA);

      it("keeps the track legible against the composer surface", () => {
        const track = composite(arc, surface, renderedTrackAlpha());

        expect(contrastRatio(track, surface)).toBeGreaterThanOrEqual(MIN_RATIO);
      });

      it("keeps the track legible under the trigger's hover fill", () => {
        const track = composite(arc, hoverFill, renderedTrackAlpha());

        expect(contrastRatio(track, hoverFill)).toBeGreaterThanOrEqual(
          MIN_RATIO,
        );
      });

      it("keeps a neutral arc distinguishable from the track in both states", () => {
        const alpha = renderedTrackAlpha();

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
