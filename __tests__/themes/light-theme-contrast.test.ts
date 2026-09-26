import { describe, expect, it } from "vitest";
import { automationActivityRowClassName } from "#/components/features/automations/automation-view-mode";
import { dropdownFilterTriggerClassName } from "#/utils/dropdown-classes";
import {
  formControlBackNavButtonClassName,
  formControlBorderClassName,
  formControlFieldClassName,
  formControlFilterTriggerClassName,
  formControlShellClassName,
} from "#/utils/form-control-classes";
import {
  LIGHT_THEMES,
  PANEL_SURFACES,
  colorUtility,
  contrast,
  over,
  surfaceColor,
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
  "filter trigger": formControlFilterTriggerClassName,
  "dropdown filter trigger": dropdownFilterTriggerClassName,
};

/** Fills that outlined controls take on hover (secondary buttons, back nav). */
const CONTROL_HOVER_FILLS = ["--oh-surface-raised", "--oh-color-tertiary"];

/** Page backgrounds controls are placed on. */
const CONTROL_PAGE_SURFACES = [
  "--oh-color-base",
  "--oh-color-base-secondary",
  "--oh-surface",
];

describe.each(LIGHT_THEMES)("%s contrast contract", (theme) => {
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

  it("makes automation and settings list-row hover visible on its list surface", () => {
    const list = surfaceColor(theme, "--oh-surface");
    const hover = over(
      utilityColor(
        theme,
        colorUtility(automationActivityRowClassName, "bg", "hover:"),
      ),
      list,
    );
    // Capped by BODY_TEXT above: a darker hover would drop row text below AA.
    expect(contrast(hover, list)).toBeGreaterThanOrEqual(1.12);
    expect(
      contrast(hover, list),
      "row hover must separate more than the raised surface it replaced",
    ).toBeGreaterThan(
      contrast(surfaceColor(theme, "--oh-surface-raised"), list),
    );
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

  it("routes every outlined shared control through the same boundary utility", () => {
    const boundary = colorUtility(formControlBorderClassName, "border");
    expect(boundary).toBe("border-border-input");
    for (const classList of [
      ...Object.values(OUTLINED_CONTROLS),
      formControlBackNavButtonClassName,
    ]) {
      expect(colorUtility(classList, "border")).toBe(boundary);
    }
  });
});
