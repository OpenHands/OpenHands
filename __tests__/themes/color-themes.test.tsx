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
    "keeps feedback, links, and keyboard focus readable on the %s theme surfaces",
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

      // Confirmation dialog descriptions are small body text on the modal surface.
      expect(
        contrast(
          luminance(resolve("--oh-modal-description-foreground")),
          modalSurface,
        ),
        "--oh-modal-description-foreground against the modal surface",
      ).toBeGreaterThanOrEqual(4.5);

      // Markdown links appear in chat, user bubbles, and file/plan previews.
      for (const surfaceRole of [
        "--oh-color-base",
        "--oh-color-base-secondary",
        "--oh-surface-raised",
        "--oh-color-tertiary",
      ]) {
        expect(
          contrast(
            luminance(resolve("--oh-link")),
            luminance(resolve(surfaceRole)),
          ),
          `--oh-link against ${surfaceRole}`,
        ).toBeGreaterThanOrEqual(4.5);

        // Single GitHub alert panels tint each Markdown host surface at 10%.
        // These are the shipped Tailwind v4 alert colors resolved to sRGB.
        for (const [alert, color] of [
          ["Note", "#2b7fff"],
          ["Tip", "#00bc7d"],
          ["Important", "#ad46ff"],
          ["Warning", "#efb100"],
          ["Caution", "#ff2056"],
        ]) {
          const host = channels(resolve(surfaceRole));
          const alertSurface = channels(color).map(
            (channel, index) => channel * 0.1 + host[index] * 0.9,
          );
          expect(
            contrast(luminance(resolve("--oh-link")), luminance(alertSurface)),
            `--oh-link against ${alert} tint over ${surfaceRole}`,
          ).toBeGreaterThanOrEqual(4.5);
        }
      }

      const statusSurfaces = ["--oh-color-base", "--oh-color-base-secondary"];
      const statusTextRoles = [
        ["--oh-feedback-success", "--oh-status-success"],
        ["--oh-feedback-error", "--oh-status-error"],
      ] as const;
      for (const surfaceToken of statusSurfaces) {
        const base = channels(resolve(surfaceToken));
        for (const [textRole, statusRole] of statusTextRoles) {
          const tint = channels(resolve(statusRole)).map(
            (channel, index) => channel * 0.05 + base[index] * 0.95,
          );
          expect(
            contrast(luminance(resolve(textRole)), luminance(tint)),
            `${textRole} against a 5% ${statusRole} tint over ${surfaceToken}`,
          ).toBeGreaterThanOrEqual(4.5);
        }
      }
      expect(
        contrast(
          luminance(resolve("--oh-contrast")),
          luminance(resolve("--oh-surface-raised")),
        ),
        "--oh-contrast against the configured credential banner surface",
      ).toBeGreaterThanOrEqual(4.5);

      // Focus contrast includes each neutral control surface.
      for (const surfaceRole of [
        "--oh-color-base",
        "--oh-color-base-secondary",
        "--oh-surface-raised",
        "--oh-interactive-hover",
      ]) {
        expect(
          contrast(
            luminance(resolve("--oh-focus")),
            luminance(resolve(surfaceRole)),
          ),
          `--oh-focus against ${surfaceRole}`,
        ).toBeGreaterThanOrEqual(3);
      }

      // @spec FF-001 — Theme-aware validation and load errors
      // Validate the automation pill and Test runs alert on their actual surfaces.
      for (const [surfaceRole, opacity] of [
        ["--oh-surface", 0.05],
        ["--oh-color-base", 0.1],
      ] as const) {
        const surface = channels(resolve(surfaceRole));
        const tintedSurface = channels(resolve("--oh-feedback-error")).map(
          (channel, index) =>
            channel * opacity + surface[index] * (1 - opacity),
        );
        expect(
          contrast(
            luminance(resolve("--oh-feedback-error")),
            luminance(tintedSurface),
          ),
          `--oh-feedback-error against its ${opacity * 100}% tint over ${surfaceRole}`,
        ).toBeGreaterThanOrEqual(4.5);
      }

      // @spec HL-001 — Home mode labels stay readable on the hover pill
      expect(
        contrast(
          luminance(resolve("--oh-contrast")),
          luminance(resolve("--oh-interactive-hover")),
        ),
        "--oh-contrast against the Home mode toggle pill",
      ).toBeGreaterThanOrEqual(4.5);

      // Context meter text and fills preserve contrast in each palette.
      const contextMeterSurface = luminance(resolve("--oh-surface-raised"));
      const contextMeterTrack = luminance(resolve("--oh-color-tertiary"));
      for (const role of ["--oh-feedback-error", "--oh-feedback-warning"]) {
        expect(
          contrast(luminance(resolve(role)), contextMeterSurface),
          `${role} against the context meter raised surface`,
        ).toBeGreaterThanOrEqual(4.5);
      }
      for (const role of [
        "--oh-foreground",
        "--oh-feedback-warning",
        "--oh-status-error",
      ]) {
        expect(
          contrast(luminance(resolve(role)), contextMeterTrack),
          `${role} against the context meter track`,
        ).toBeGreaterThanOrEqual(3);
      }

      // The composer popover outlines the full range even when its tertiary
      // track sits close to the raised background. The button background paints
      // under its translucent border; compare that composite with the popover.
      const boundaryWeight =
        Number.parseFloat(resolve("--oh-context-window-track-weight")) / 100;
      const trackBackground = channels(resolve("--oh-color-tertiary"));
      const popoverBoundary = channels(
        resolve("--oh-context-window-foreground"),
      ).map(
        (channel, index) =>
          channel * boundaryWeight +
          trackBackground[index] * (1 - boundaryWeight),
      );
      expect(
        contrast(luminance(popoverBoundary), contextMeterSurface),
        "context range boundary against the composer popover raised surface",
      ).toBeGreaterThanOrEqual(3);

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

      // Critic scores and small supporting text render directly on the chat canvas.
      for (const role of [
        "--oh-feedback-success",
        "--oh-feedback-warning",
        "--oh-feedback-error",
        "--oh-text-tertiary",
        "--oh-contrast",
      ]) {
        expect(
          contrast(
            luminance(resolve(role)),
            luminance(resolve("--oh-color-base")),
          ),
          `${role} against the critic canvas surface`,
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
