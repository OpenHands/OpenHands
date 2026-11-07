import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentServerUIRoot } from "#/components/providers/agent-server-ui-root";
import {
  AVAILABLE_COLOR_THEMES,
  COLOR_THEME_BOOTSTRAP_SCRIPT,
  DEFAULT_COLOR_THEME,
  applyColorTheme,
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
});
