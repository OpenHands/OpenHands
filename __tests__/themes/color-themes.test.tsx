import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { AgentServerUIRoot } from "#/components/providers/agent-server-ui-root";
import { AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES } from "#/styles/agent-server-ui-style-scope";
import {
  AVAILABLE_COLOR_THEMES,
  COLOR_THEMES,
  applyColorTheme,
} from "#/themes/color-themes";

describe("color themes", () => {
  // @spec MCP-004 — Form feedback follows readable theme text roles
  it.each(Object.entries(COLOR_THEMES))(
    "keeps small form feedback readable on the %s modal surface",
    (_key, theme) => {
      // Resolve the shipped palette, including stylesheet-owned defaults.
      const variables: Record<string, string> = {
        ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
        ...theme.scale,
        ...theme.tokens,
      };
      const resolve = (name: string): string => {
        const value = variables[name];
        expect(value, `Missing theme token ${name}`).toBeDefined();
        const reference = /^var\((--[\w-]+)\)$/.exec(value);
        return reference ? resolve(reference[1]) : value;
      };
      const luminance = (hex: string): number => {
        const channels = [1, 3, 5].map((offset) => {
          const channel =
            Number.parseInt(hex.slice(offset, offset + 2), 16) / 255;
          return channel <= 0.04045
            ? channel / 12.92
            : ((channel + 0.055) / 1.055) ** 2.4;
        });
        return (
          channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
        );
      };
      const surface = luminance(resolve("--oh-color-base-secondary"));

      for (const role of [
        "--oh-feedback-error",
        "--oh-feedback-success",
        "--oh-feedback-warning",
      ]) {
        const ink = luminance(resolve(role));
        const ratio =
          (Math.max(ink, surface) + 0.05) / (Math.min(ink, surface) + 0.05);
        expect(
          ratio,
          `${role} against the modal surface`,
        ).toBeGreaterThanOrEqual(4.5);
      }
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
  it("preserves host feedback colors when the selected palette changes", () => {
    const overrides = {
      "--oh-feedback-error": "#672727",
      "--oh-feedback-success": "#24542d",
      "--oh-feedback-warning": "#895500",
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

  // @spec MCP-004 — Warning feedback text meets 4.5:1 on real built-in surfaces
  it.each(Object.entries(COLOR_THEMES))(
    "keeps feedback-warning readable across built-in surfaces in %s",
    (_key, theme) => {
      const variables: Record<string, string> = {
        ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
        ...theme.scale,
        ...theme.tokens,
      };
      const resolve = (name: string): string => {
        const value =
          variables[name] ?? (name === "--oh-warning" ? "#c9b974" : undefined);
        expect(value, `Missing theme token ${name}`).toBeDefined();
        const reference = /^var\((--[\w-]+)\)$/.exec(value!);
        return reference ? resolve(reference[1]) : value!;
      };
      const hexToRgb = (hex: string): [number, number, number] => [
        Number.parseInt(hex.slice(1, 3), 16) / 255,
        Number.parseInt(hex.slice(3, 5), 16) / 255,
        Number.parseInt(hex.slice(5, 7), 16) / 255,
      ];
      const rgbToLuminance = (rgb: [number, number, number]): number => {
        const channels = rgb.map((c) =>
          c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4,
        );
        return (
          channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
        );
      };
      const contrast = (l1: number, l2: number) =>
        (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);

      const warningTextHex = resolve("--oh-feedback-warning");
      const warningTextLum = rgbToLuminance(hexToRgb(warningTextHex));

      // 1. Home banner surface (bg-surface-raised)
      const surfaceRaisedHex = resolve("--oh-surface-raised");
      const surfaceRaisedLum = rgbToLuminance(hexToRgb(surfaceRaisedHex));
      expect(
        contrast(warningTextLum, surfaceRaisedLum),
        `--oh-feedback-warning against surface-raised in ${_key}`,
      ).toBeGreaterThanOrEqual(4.5);

      // 2. Modal / credential surface (bg-base-secondary)
      const baseSecondaryHex = resolve("--oh-color-base-secondary");
      const baseSecondaryLum = rgbToLuminance(hexToRgb(baseSecondaryHex));
      expect(
        contrast(warningTextLum, baseSecondaryLum),
        `--oh-feedback-warning against base-secondary in ${_key}`,
      ).toBeGreaterThanOrEqual(4.5);

      // 3. Trust panel (5% warning tint blended over page base background)
      const baseHex = resolve("--oh-color-base");
      const warningControlHex = resolve("--oh-warning");
      const baseRgb = hexToRgb(baseHex);
      const warningControlRgb = hexToRgb(warningControlHex);
      const trustPanelRgb: [number, number, number] = [
        warningControlRgb[0] * 0.05 + baseRgb[0] * 0.95,
        warningControlRgb[1] * 0.05 + baseRgb[1] * 0.95,
        warningControlRgb[2] * 0.05 + baseRgb[2] * 0.95,
      ];
      const trustPanelLum = rgbToLuminance(trustPanelRgb);
      expect(
        contrast(warningTextLum, trustPanelLum),
        `--oh-feedback-warning against trust panel in ${_key}`,
      ).toBeGreaterThanOrEqual(4.5);
    },
  );
});
