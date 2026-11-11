import { describe, expect, it } from "vitest";
import { automationActivityRowClassName } from "#/components/features/automations/automation-view-mode";
import { dropdownFilterTriggerClassName } from "#/utils/dropdown-classes";
import {
  formControlBackNavButtonClassName,
  formControlBorderClassName,
  formControlFieldClassName,
  formControlShellClassName,
} from "#/utils/form-control-classes";
import {
  ALL_THEMES,
  LIGHT_THEMES,
  PANEL_SURFACES,
  colorUtility,
  contrast,
  over,
  surfaceColor,
  tokenColor,
  utilityColor,
} from "./theme-color-resolver";

const BODY_TEXT = [
  "--oh-foreground",
  "--oh-color-content",
  "--oh-muted",
  "--oh-text-tertiary",
  "--oh-text-dim",
  "--oh-link",
  "--oh-info",
  "--oh-success",
  "--oh-warning",
  "--oh-danger",
];

/** Shared controls that draw an interactive boundary on a filled shell. */
const OUTLINED_CONTROLS = {
  field: formControlFieldClassName,
  shell: formControlShellClassName,
  "dropdown filter trigger": dropdownFilterTriggerClassName,
};

/** Fills that outlined controls take on hover (secondary buttons, back nav). */
const CONTROL_HOVER_FILLS = ["--oh-surface-raised", "--oh-color-tertiary"];

/**
 * Page backgrounds controls are placed on, including the deepest panel and the
 * selected-row fill, where an input edge has the least room above 3:1.
 */
const CONTROL_PAGE_SURFACES = [
  "--oh-color-base",
  "--oh-color-base-secondary",
  "--oh-surface",
  "--oh-surface-deep",
  "--oh-interactive-selected",
];

describe.each(["light-plus", "solarized-light", "paper"] as const)(
  "%s text contrast contract",
  (theme) => {
    it.each(BODY_TEXT)("%s meets WCAG AA on every panel surface", (text) => {
      for (const surface of PANEL_SURFACES) {
        expect
          .soft(
            contrast(surfaceColor(theme, text), surfaceColor(theme, surface)),
            `${text} on ${surface}`,
          )
          .toBeGreaterThanOrEqual(4.5);
      }
    });
  },
);

// These surface choices belong to the Light+/Solarized contrast pass. Paper
// keeps its own flatter surface hierarchy.
const CONTRAST_PASS_THEMES = ["light-plus", "solarized-light"] as const;

describe.each(CONTRAST_PASS_THEMES)("%s palette contrast contract", (theme) => {
  it("makes hover rows visible on menus, sidebars, and cards", () => {
    const hover = surfaceColor(theme, "--oh-interactive-hover");
    for (const surface of [
      "--oh-color-base",
      "--oh-surface",
      "--oh-color-tertiary",
    ]) {
      expect
        .soft(
          contrast(hover, surfaceColor(theme, surface)),
          `hover on ${surface}`,
        )
        .toBeGreaterThanOrEqual(1.2);
    }
  });

  it("separates raised buttons and dividers from the page", () => {
    const base = surfaceColor(theme, "--oh-color-base");
    const surface = surfaceColor(theme, "--oh-surface");
    expect(
      contrast(surfaceColor(theme, "--oh-surface-raised"), base),
    ).toBeGreaterThanOrEqual(1.2);
    expect(
      contrast(surfaceColor(theme, "--oh-border-subtle"), surface),
    ).toBeGreaterThanOrEqual(1.1);
    expect(
      contrast(surfaceColor(theme, "--oh-border"), base),
    ).toBeGreaterThanOrEqual(1.5);
  });
});

it("keeps Light+ backend hosts and uploaded-file details readable on interactive fills", () => {
  // BackendRow keeps muted host text on hover and keyboard focus. UploadedFile
  // uses the same pairing for its extension and its filename while uploading.
  const background = surfaceColor("light-plus", "--oh-interactive-hover");
  const text = utilityColor("light-plus", "text-muted");

  expect(contrast(over(text, background), background)).toBeGreaterThanOrEqual(
    4.5,
  );
});

describe.each(LIGHT_THEMES)("%s shared-control contrast", (theme) => {
  it("keeps filled-button labels readable", () => {
    for (const [fill, label] of [
      ["--oh-color-primary", "--oh-accent-foreground"],
      ["--oh-accent", "--oh-accent-foreground"],
      ["--oh-success", "--oh-success-foreground"],
      ["--oh-warning", "--oh-warning-foreground"],
      ["--oh-danger", "--oh-danger-foreground"],
    ]) {
      expect
        .soft(
          contrast(surfaceColor(theme, fill), surfaceColor(theme, label)),
          `${label} on ${fill}`,
        )
        .toBeGreaterThanOrEqual(4.5);
    }
  });

  it("makes automation row hover and keyboard focus visible on its list surface", () => {
    const list = surfaceColor(theme, "--oh-surface");
    for (const state of ["hover:", "has-[:focus-visible]:"]) {
      const variant =
        theme === "paper"
          ? `[[data-agent-server-ui][data-color-theme=paper]_&]:${state}`
          : state;
      const hover = over(
        utilityColor(
          theme,
          colorUtility(automationActivityRowClassName, "bg", variant),
        ),
        list,
      );
      expect(contrast(hover, list), state).toBeGreaterThanOrEqual(1.12);
      if (theme === "light-plus" || theme === "solarized-light") {
        // These palettes darken the old raised treatment; Paper preserves it.
        expect(
          contrast(hover, list),
          "row hover must separate more than the raised surface it replaced",
        ).toBeGreaterThan(
          contrast(surfaceColor(theme, "--oh-surface-raised"), list),
        );
      }
    }
  });

  it("keeps the established Neutral hover treatment as a dark override", () => {
    expect(automationActivityRowClassName).toContain(
      "dark:hover:bg-surface-raised",
    );
    expect(automationActivityRowClassName).toContain(
      "dark:has-[:focus-visible]:bg-surface-raised",
    );
  });

  it.each(Object.entries(OUTLINED_CONTROLS))(
    "outlines the %s at the WCAG 3:1 non-text minimum against its fill, page, and hover fill",
    (_, classList) => {
      const border = utilityColor(theme, colorUtility(classList, "border"));
      const fill = utilityColor(theme, colorUtility(classList, "bg"));
      for (const page of CONTROL_PAGE_SURFACES) {
        const pageColor = surfaceColor(theme, page);
        const fillColor = over(fill, pageColor);
        expect
          .soft(
            contrast(over(border, fillColor), fillColor),
            `vs fill on ${page}`,
          )
          .toBeGreaterThanOrEqual(3);
        expect
          .soft(contrast(over(border, pageColor), pageColor), `vs ${page}`)
          .toBeGreaterThanOrEqual(3);
      }
      for (const hoverFill of CONTROL_HOVER_FILLS) {
        const hoverColor = surfaceColor(theme, hoverFill);
        expect
          .soft(
            contrast(over(border, hoverColor), hoverColor),
            `vs ${hoverFill}`,
          )
          .toBeGreaterThanOrEqual(3);
      }
    },
  );

  it("draws row dividers that read as separators on every panel surface", () => {
    const divider = utilityColor(theme, "border-divider");
    for (const surface of PANEL_SURFACES) {
      const backdrop = surfaceColor(theme, surface);
      const painted = contrast(over(divider, backdrop), backdrop);
      expect.soft(painted, `divider on ${surface}`).toBeGreaterThanOrEqual(1.5);
      expect
        .soft(painted, `divider must out-separate border-subtle on ${surface}`)
        .toBeGreaterThan(
          contrast(surfaceColor(theme, "--oh-border-subtle"), backdrop),
        );
    }
  });

  it("keeps danger menu text readable on overlays and panel surfaces", () => {
    const danger = utilityColor(theme, "text-danger");
    for (const surface of [...PANEL_SURFACES, "--oh-overlay"]) {
      const backdrop = surfaceColor(theme, surface);
      expect
        .soft(contrast(over(danger, backdrop), backdrop), `on ${surface}`)
        .toBeGreaterThanOrEqual(4.5);
    }
  });

  it("gives every outlined shared control the same light boundary color", () => {
    const boundary = tokenColor(theme, "--oh-border-input");
    for (const classList of [
      formControlBorderClassName,
      ...Object.values(OUTLINED_CONTROLS),
      formControlBackNavButtonClassName,
    ]) {
      expect(utilityColor(theme, colorUtility(classList, "border"))).toEqual(
        boundary,
      );
    }
  });
});

describe.each(ALL_THEMES.filter((key) => !LIGHT_THEMES.includes(key)))(
  "%s divider parity",
  (theme) => {
    it("keeps dividers on the established border-subtle color", () => {
      expect(utilityColor(theme, "border-divider")).toEqual(
        tokenColor(theme, "--oh-border-subtle"),
      );
    });
  },
);
