import { act, render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it } from "vitest";
import { AgentServerUIRoot } from "#/components/providers/agent-server-ui-root";
import { AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES } from "#/styles/agent-server-ui-style-scope";
import {
  AVAILABLE_COLOR_THEMES,
  COLOR_THEMES,
  applyColorTheme,
} from "#/themes/color-themes";

const defaultWarningColor = readFileSync(
  `${process.cwd()}/src/tailwind.css`,
  "utf-8",
).match(/^\s*--oh-warning:\s*([^;]+);/m)?.[1];

describe("color themes", () => {
  // @spec MCP-004 — Form feedback follows readable theme text roles
  // @spec WN-001 — Warning feedback also meets the small-text contrast contract
  it.each(Object.entries(COLOR_THEMES))(
    "keeps small text readable on the %s theme surfaces",
    (_key, theme) => {
      // Resolve the shipped palette, including stylesheet-owned defaults.
      const variables: Record<string, string> = {
        ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
        ...theme.scale,
        ...theme.tokens,
      };
      // Brand defaults stay CSS-owned so runtime theme overrides can win.
      if (defaultWarningColor) {
        variables["--oh-warning"] ??= defaultWarningColor;
      }
      const resolve = (name: string): string => {
        const value = variables[name];
        expect(value, `Missing theme token ${name}`).toBeDefined();
        const reference = /^var\((--[\w-]+)\)$/.exec(value);
        return reference ? resolve(reference[1]) : value;
      };
      const channels = (hex: string): number[] =>
        [1, 3, 5].map(
          (offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255,
        );
      const luminance = (color: string | number[]): number => {
        const values = typeof color === "string" ? channels(color) : color;
        const linear = values.map((channel) =>
          channel <= 0.04045
            ? channel / 12.92
            : ((channel + 0.055) / 1.055) ** 2.4,
        );
        return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
      };
      const contrast = (ink: number, surface: number) =>
        (Math.max(ink, surface) + 0.05) / (Math.min(ink, surface) + 0.05);
      const modalSurface = luminance(resolve("--oh-color-base-secondary"));

      for (const role of ["--oh-feedback-error", "--oh-feedback-success"]) {
        expect(
          contrast(luminance(resolve(role)), modalSurface),
          `${role} against the modal surface`,
        ).toBeGreaterThanOrEqual(4.5);
      }

      // @spec FF-001 — Small error text stays readable on tinted draft error pills
      const errorInk = luminance(resolve("--oh-feedback-error"));
      const draftBarSurface = channels(resolve("--oh-surface"));
      const errorPillTint = channels("#fb2c36");
      const errorPillSurface = errorPillTint.map(
        (channel, index) => channel * 0.05 + draftBarSurface[index] * 0.95,
      );
      expect(
        contrast(errorInk, luminance(errorPillSurface)),
        "--oh-feedback-error against the draft validity error pill",
      ).toBeGreaterThanOrEqual(4.5);

      const alertBackground = channels(resolve("--oh-background"));
      const alertSurface = errorPillTint.map(
        (channel, index) => channel * 0.1 + alertBackground[index] * 0.9,
      );
      expect(
        contrast(errorInk, luminance(alertSurface)),
        "--oh-feedback-error against the draft runs error alert",
      ).toBeGreaterThanOrEqual(4.5);

      // @spec HL-001 — Home mode labels stay readable on the hover pill
      expect(
        contrast(
          luminance(resolve("--oh-contrast")),
          luminance(resolve("--oh-interactive-hover")),
        ),
        "--oh-contrast against the Home mode toggle pill",
      ).toBeGreaterThanOrEqual(4.5);

      const warningInk = luminance(resolve("--oh-feedback-warning"));
      const trustBackground = channels(resolve("--oh-background"));
      const warningTint = channels(resolve("--oh-warning"));
      const tintedTrustSurface = warningTint.map(
        (channel, index) => channel * 0.05 + trustBackground[index] * 0.95,
      );
      const warningSurfaces: Array<[string, number]> = [
        ["ACP credential surface", modalSurface],
        ["Home raised banner", luminance(resolve("--oh-surface-raised"))],
        ["Apps trust notice tint", luminance(tintedTrustSurface)],
      ];

      for (const [surfaceName, surface] of warningSurfaces) {
        expect(
          contrast(warningInk, surface),
          `--oh-feedback-warning against ${surfaceName}`,
        ).toBeGreaterThanOrEqual(4.5);
      }

      // @spec TN-001 — Toast text stays readable on the toast surface
      expect(
        contrast(
          luminance(resolve("--oh-contrast")),
          luminance(resolve("--oh-color-tertiary")),
        ),
        "--oh-contrast against the toast surface",
      ).toBeGreaterThanOrEqual(4.5);
    },
  );

  afterEach(() => {
    document
      .querySelectorAll("style[data-theme-test]")
      .forEach((el) => el.remove());
  });

  it("leaves omitted tokens owned by consumer stylesheets", () => {
    const sheet = document.createElement("style");
    sheet.dataset.themeTest = "";
    sheet.textContent =
      "[data-agent-server-ui] { --oh-color-primary: #123456; --oh-radius: 12px; }";
    document.head.appendChild(sheet);
    render(
      <AgentServerUIRoot data-testid="consumer-scope">
        Canvas
      </AgentServerUIRoot>,
    );
    act(() => applyColorTheme("openhands-neutral"));
    const style = getComputedStyle(screen.getByTestId("consumer-scope"));
    expect(style.getPropertyValue("--oh-color-primary")).toBe("#123456");
    expect(style.getPropertyValue("--oh-radius")).toBe("12px");
  });
  // @spec MCP-004 — Form feedback follows readable theme text roles
  // @spec WN-001 — Warning feedback can be overridden by an embedding host
  it("preserves host feedback colors when the selected palette changes", () => {
    const overrides = {
      "--oh-feedback-error": "#672727",
      "--oh-feedback-success": "#24542d",
      "--oh-feedback-warning": "#755300",
    } as const;
    render(
      <AgentServerUIRoot
        data-testid="feedback-scope"
        styleOverrides={overrides}
      >
        Canvas
      </AgentServerUIRoot>,
    );

    act(() => applyColorTheme("light-plus"));

    const style = getComputedStyle(screen.getByTestId("feedback-scope"));
    for (const [token, value] of Object.entries(overrides)) {
      expect(style.getPropertyValue(token)).toBe(value);
    }
    document.getElementById("oh-color-theme-override")?.remove();
  });
  it("includes OpenHands-Neo as a neutral-based theme with white button tokens", () => {
    const neo = COLOR_THEMES["openhands-neo"];

    expect(neo.label).toBe("OpenHands-Neo");
    expect(neo.scale).toEqual(COLOR_THEMES["openhands-neutral"].scale);
    expect(neo.heroui).toEqual(COLOR_THEMES["openhands-neutral"].heroui);
    expect(neo.tokens?.["--oh-color-primary"]).toBe("#ffffff");
    expect(neo.tokens?.["--oh-accent"]).toBe("#ffffff");
  });

  it("exposes Neo in the settings theme picker", () => {
    expect(AVAILABLE_COLOR_THEMES.map((theme) => theme.key)).toContain(
      "openhands-neo",
    );
    expect(
      AVAILABLE_COLOR_THEMES.find((theme) => theme.key === "openhands-neo")
        ?.label,
    ).toBe("OpenHands-Neo");
  });

  it("injects white primary tokens when applying OpenHands-Neo", () => {
    document.body.setAttribute("data-agent-server-ui", "");

    applyColorTheme("openhands-neo");

    const styleEl = document.getElementById("oh-color-theme-override");
    expect(styleEl?.textContent).toContain("--oh-color-primary: #ffffff;");
    expect(styleEl?.textContent).toContain("--oh-accent: #ffffff;");

    styleEl?.remove();
    document.body.removeAttribute("data-agent-server-ui");
    document.body.style.removeProperty("--oh-color-primary");
    document.body.style.removeProperty("--oh-accent");
    document.body.style.removeProperty("--oh-warning");
  });

  it("injects override rules with order-independent doubled scope selectors", () => {
    // Act
    applyColorTheme("openhands-neutral");

    // Assert: doubled selectors (0,2,0) out-specify the base sheet's unlayered
    // [data-agent-server-ui] variable rules (0,1,0), so the override wins even
    // when React 19 re-inserts the base stylesheet <link> after this tag.
    const styleEl = document.getElementById("oh-color-theme-override");
    expect(styleEl?.textContent).toContain(
      "[data-agent-server-ui][data-agent-server-ui] {",
    );
    expect(styleEl?.textContent).toContain(
      "[data-agent-server-ui] [data-theme][data-theme] {",
    );

    styleEl?.remove();
  });

  it("re-appends the override style tag to the end of <head> on every apply", () => {
    // Arrange: first apply creates the tag, then a later stylesheet lands
    // after it (as React 19 does with the base CSS <link> in the built SPA).
    applyColorTheme("openhands-neutral");
    const laterSheet = document.createElement("style");
    document.head.appendChild(laterSheet);

    // Act
    applyColorTheme("openhands-deepsea");

    // Assert
    expect(document.head.lastElementChild?.id).toBe("oh-color-theme-override");

    laterSheet.remove();
    document.getElementById("oh-color-theme-override")?.remove();
  });

  it("applies Neo button tokens on the scoped UI root used by primary buttons", () => {
    const baseSheet = document.createElement("style");
    baseSheet.dataset.themeTest = "";
    baseSheet.textContent =
      "[data-agent-server-ui] { --oh-color-primary: #c9b974; }";
    document.head.appendChild(baseSheet);
    render(
      <AgentServerUIRoot>
        <button type="button" data-testid="primary-button">
          Save
        </button>
      </AgentServerUIRoot>,
    );

    act(() => applyColorTheme("openhands-neo"));

    const scopeRoot = screen
      .getByTestId("primary-button")
      .closest("[data-agent-server-ui]") as HTMLElement;

    expect(
      getComputedStyle(scopeRoot).getPropertyValue("--oh-color-primary"),
    ).toBe("#ffffff");

    act(() => applyColorTheme("openhands-neutral"));

    expect(
      getComputedStyle(scopeRoot).getPropertyValue("--oh-color-primary"),
    ).toBe("#c9b974");
  });
});
