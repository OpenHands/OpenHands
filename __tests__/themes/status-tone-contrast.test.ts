import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  statusToneBadgeClassName,
  statusToneBannerClassName,
  type StatusTone,
} from "#/utils/status-tone-classes";
import {
  ALL_THEMES,
  PANEL_SURFACES,
  colorUtility,
  contrast,
  over,
  surfaceColor,
  utilityColor,
} from "./theme-color-resolver";

const TONES: StatusTone[] = ["success", "warning", "danger", "info"];

const HELPERS = {
  badge: statusToneBadgeClassName,
  banner: statusToneBannerClassName,
};

describe.each(ALL_THEMES)("%s status badges", (theme) => {
  describe.each(Object.entries(HELPERS))("%s helper", (_, helper) => {
    it.each(TONES)(
      "%s ink clears 4.5:1 on its fill composited over every panel surface",
      (tone) => {
        const fill = utilityColor(theme, colorUtility(helper[tone], "bg"));
        const ink = utilityColor(theme, colorUtility(helper[tone], "text"));
        for (const surface of PANEL_SURFACES) {
          const badge = over(fill, surfaceColor(theme, surface));
          expect
            .soft(contrast(over(ink, badge), badge), `${tone} on ${surface}`)
            .toBeGreaterThanOrEqual(4.5);
        }
      },
    );
  });
});

const SRC_ROOT = resolve(__dirname, "../../src");

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.tsx?$/.test(entry) ? [path] : [];
  });
}

describe("status tone usage", () => {
  it("does not hand-roll translucent tone fills under same-tone text", () => {
    const tone =
      "(success|warning|danger|info|semantic-success|semantic-danger|status-success|status-error)";
    const fill = new RegExp(`\\bbg-${tone}/\\d+`, "g");
    const offenders = new Set<string>();
    for (const file of sourceFiles(SRC_ROOT)) {
      for (const [literal] of readFileSync(file, "utf8").matchAll(
        /"[^"\n]*"/g,
      )) {
        for (const [, fillTone] of literal.matchAll(fill)) {
          const bare = fillTone.replace(/^(semantic|status)-/, "");
          if (new RegExp(`\\btext-(\\w+-)?${bare}\\b`).test(literal)) {
            offenders.add(`${relative(SRC_ROOT, file)}: ${literal}`);
          }
        }
      }
    }
    expect([...offenders]).toEqual([]);
  });
});
