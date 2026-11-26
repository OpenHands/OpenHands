import { describe, expect, it } from "vitest";
import { PANEL_SURFACES, contrast, surfaceColor } from "./theme-color-resolver";

const DARK_THEMES = [
  "openhands-neutral",
  "openhands-neo",
  "openhands-deepsea",
] as const;

describe.each(DARK_THEMES)("%s muted text contrast contract", (theme) => {
  it("--oh-muted meets WCAG AA on every panel surface", () => {
    for (const surface of PANEL_SURFACES) {
      expect
        .soft(
          contrast(
            surfaceColor(theme, "--oh-muted"),
            surfaceColor(theme, surface),
          ),
          `--oh-muted on ${surface}`,
        )
        .toBeGreaterThanOrEqual(4.5);
    }
  });
});
