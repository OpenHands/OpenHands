import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentServerUIRoot } from "#/components/providers/agent-server-ui-root";
import {
  AVAILABLE_COLOR_THEMES,
  COLOR_THEME_BOOTSTRAP_SCRIPT,
  DEFAULT_COLOR_THEME,
  applyColorTheme,
  getColorThemeBaseColor,
  getColorThemeCss,
  readPersistedColorTheme,
  setColorTheme,
  subscribeColorTheme,
} from "#/themes/color-themes";

describe("color themes", () => {
  beforeEach(() => {
    window.localStorage.clear();
    applyColorTheme(DEFAULT_COLOR_THEME);
  });
  afterEach(cleanup);

  it("offers both light palettes", () => {
    expect(AVAILABLE_COLOR_THEMES).toEqual(
      expect.arrayContaining([
        { key: "light-plus", label: "Light+" },
        { key: "solarized-light", label: "Solarized Light" },
      ]),
    );
  });

  it("reactively updates mounted and newly mounted roots without overwriting caller styles", () => {
    const props = {
      styleOverrides: {
        "--oh-background": "#123456",
        "--oh-color-primary": "#abcdef",
      },
    };
    const { rerender } = render(
      <AgentServerUIRoot {...props} data-testid="scope">
        Canvas
      </AgentServerUIRoot>,
    );
    act(() => setColorTheme("light-plus"));
    const scope = screen.getByTestId("scope");
    expect(scope).toHaveAttribute("data-color-theme", "light-plus");
    expect(scope.firstElementChild).toHaveClass("light");
    expect(scope.style.getPropertyValue("--oh-background")).toBe("#123456");
    expect(scope.style.getPropertyValue("--oh-color-primary")).toBe("#abcdef");
    rerender(
      <AgentServerUIRoot {...props} data-testid="scope">
        Updated
      </AgentServerUIRoot>,
    );
    expect(scope.style.getPropertyValue("--oh-background")).toBe("#123456");
    render(<AgentServerUIRoot data-testid="new-scope">New</AgentServerUIRoot>);
    expect(screen.getByTestId("new-scope")).toHaveAttribute(
      "data-color-scheme",
      "light",
    );
    expect(
      screen.getByTestId("new-scope").style.getPropertyValue("--oh-background"),
    ).toBe("");
    act(() => setColorTheme("openhands-neutral"));
    expect(scope.firstElementChild).toHaveClass("dark");
    expect(scope.style.getPropertyValue("--oh-background")).toBe("#123456");
    expect(
      document.getElementById("oh-color-theme-override")?.textContent,
    ).not.toContain("--oh-background: #FFFFFF");
  });

  it("persists selection and notifies only after CSS is applied", () => {
    const listener = vi.fn(() => {
      expect(
        document.getElementById("oh-color-theme-override")?.textContent,
      ).toBe(getColorThemeCss("solarized-light"));
    });
    const unsubscribe = subscribeColorTheme(listener);
    setColorTheme("solarized-light");
    expect(readPersistedColorTheme()).toBe("solarized-light");
    expect(listener).toHaveBeenCalledOnce();
    unsubscribe();
  });

  it.each(["light-plus", "solarized-light", "openhands-neutral"] as const)(
    "bootstraps %s before React mounts",
    (key) => {
      // Reproduce the prerendered dark wrapper while app initialization waits.
      render(<AgentServerUIRoot data-testid="shell">Canvas</AgentServerUIRoot>);
      const wrapper = screen.getByTestId("shell").firstElementChild!;
      expect(wrapper).toHaveAttribute("data-theme", "dark");
      localStorage.setItem("openhands-color-theme", key);
      document.getElementById("oh-color-theme-override")?.remove();
      window.eval(COLOR_THEME_BOOTSTRAP_SCRIPT);
      expect(
        document.getElementById("oh-color-theme-override")?.textContent,
      ).toBe(getColorThemeCss(key));
      expect(document.documentElement.style.colorScheme).toBe(
        key === "openhands-neutral" ? "dark" : "light",
      );
      // A later base sheet must not restore HeroUI's dark native controls.
      const heroSheet = document.createElement("style");
      heroSheet.textContent = ".dark { color-scheme: dark; }";
      document.head.appendChild(heroSheet);
      try {
        expect(getComputedStyle(wrapper).colorScheme).toBe(
          key === "openhands-neutral" ? "dark" : "light",
        );
      } finally {
        heroSheet.remove();
      }
    },
  );

  it.each(["missing-theme", "constructor", "__proto__"])(
    "rejects invalid stored key %s",
    (key) => {
      localStorage.setItem("openhands-color-theme", key);
      expect(readPersistedColorTheme()).toBe(DEFAULT_COLOR_THEME);
      document.getElementById("oh-color-theme-override")?.remove();
      expect(() => window.eval(COLOR_THEME_BOOTSTRAP_SCRIPT)).not.toThrow();
      expect(
        document.getElementById("oh-color-theme-override")?.textContent,
      ).toBe(getColorThemeCss(DEFAULT_COLOR_THEME));
    },
  );

  // An installed window takes its title bar and task-switcher color from this
  // tag, so a stale value leaves the OS chrome a different color than the app.
  describe("theme-color meta", () => {
    const readThemeColor = () =>
      document
        .querySelector<HTMLMetaElement>('meta[name="theme-color"]')
        ?.getAttribute("content");

    it.each([
      ["openhands-neutral", "#181818"],
      ["openhands-deepsea", "#0B0E14"],
      ["light-plus", "#FFFFFF"],
      ["solarized-light", "#FDF6E3"],
    ] as const)("resolves %s to its painted base color", (key, expected) => {
      expect(getColorThemeBaseColor(key)).toBe(expected);
    });

    it("follows the selected palette", () => {
      act(() => setColorTheme("light-plus"));
      expect(readThemeColor()).toBe(getColorThemeBaseColor("light-plus"));

      act(() => setColorTheme("openhands-deepsea"));
      expect(readThemeColor()).toBe(
        getColorThemeBaseColor("openhands-deepsea"),
      );
    });

    it("reuses the existing tag rather than stacking duplicates", () => {
      act(() => setColorTheme("light-plus"));
      act(() => setColorTheme("solarized-light"));

      expect(
        document.querySelectorAll('meta[name="theme-color"]'),
      ).toHaveLength(1);
    });

    // The bootstrap runs in <head> before first paint; leaving the tag to React
    // would flash the manifest's color while the app hydrates.
    it("is set by the bootstrap script before paint", () => {
      localStorage.setItem("openhands-color-theme", "solarized-light");
      document.querySelector('meta[name="theme-color"]')?.remove();

      window.eval(COLOR_THEME_BOOTSTRAP_SCRIPT);

      expect(readThemeColor()).toBe(getColorThemeBaseColor("solarized-light"));
    });
  });
});
