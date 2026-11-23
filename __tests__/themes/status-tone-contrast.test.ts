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
  LIGHT_THEMES,
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

describe.each(LIGHT_THEMES)("%s feedback on status surfaces", (theme) => {
  it.each([
    ["success", "text-feedback-success"],
    ["danger", "text-feedback-error"],
  ] as const)(
    "%s feedback clears 4.5:1 on the opaque status fill",
    (tone, feedback) => {
      const fill = utilityColor(
        theme,
        colorUtility(statusToneBannerClassName[tone], "bg"),
      );
      const ink = utilityColor(theme, feedback);
      for (const surface of PANEL_SURFACES) {
        const banner = over(fill, surfaceColor(theme, surface));
        expect
          .soft(
            contrast(over(ink, banner), banner),
            `${feedback} on ${surface}`,
          )
          .toBeGreaterThanOrEqual(4.5);
      }
    },
  );
});

const SRC_ROOT = resolve(__dirname, "../../src");

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.tsx?$/.test(entry) ? [path] : [];
  });
}

function hasTranslucentTonePair(classList: string): boolean {
  const tone =
    "(success|warning|danger|info|semantic-success|semantic-danger|status-success|status-error)";
  const fill = new RegExp(`^bg-(?:${tone}|\\[var\\(--oh-${tone}\\)\\])/\\d+$`);
  const classes = classList.split(/\s+/);
  return classes.some((className) => {
    const match = fill.exec(className);
    if (!match) return false;
    const bare = (match[1] ?? match[2]).replace(/^(semantic|status)-/, "");
    const ink = new RegExp(
      `^text-(?:(?:[\\w-]+-)?${bare}|\\[var\\(--oh-(?:[\\w-]+-)?${bare}\\)\\])$`,
    );
    return classes.some((candidate) => ink.test(candidate));
  });
}

describe("status tone usage", () => {
  it.each([
    "bg-warning/15 text-warning",
    "bg-status-success/10 text-feedback-success",
    "bg-[var(--oh-warning)]/15 text-[var(--oh-warning)]",
    "bg-warning/15 text-[var(--oh-warning)]",
    "bg-[var(--oh-warning)]/15 text-warning",
  ])("detects an unprefixed translucent tone pair: %s", (classes) => {
    expect(hasTranslucentTonePair(classes)).toBe(true);
  });

  it.each([
    "bg-warning-soft text-warning-soft-foreground",
    "dark:bg-warning/15 dark:text-warning",
    "dark:bg-[var(--oh-warning)]/15 dark:text-[var(--oh-warning)]",
    "hover:bg-warning/15 hover:text-warning",
    "bg-warning/15 dark:text-warning",
    "bg-warning/15 text-info",
  ])(
    "allows semantic fills and variant-only compatibility styles: %s",
    (classes) => {
      expect(hasTranslucentTonePair(classes)).toBe(false);
    },
  );

  it("does not hand-roll translucent tone fills for the light-theme defaults", () => {
    // Variant-only overrides preserve the established dark appearance. Both
    // named utilities and arbitrary var() utilities must obey the light rule.
    const offenders = new Set<string>();
    for (const file of sourceFiles(SRC_ROOT)) {
      for (const [literal] of readFileSync(file, "utf8").matchAll(
        /"[^"\n]*"/g,
      )) {
        if (hasTranslucentTonePair(literal.slice(1, -1))) {
          offenders.add(`${relative(SRC_ROOT, file)}: ${literal}`);
        }
      }
    }
    expect([...offenders]).toEqual([]);
  });
});
