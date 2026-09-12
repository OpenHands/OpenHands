import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AVAILABLE_COLOR_THEMES,
  applyColorTheme,
  readPersistedColorTheme,
  setColorTheme,
  subscribeColorTheme,
} from "#/themes/color-themes";

function renderThemeScope() {
  document.body.innerHTML = `
    <div data-agent-server-ui>
      <div class="dark" data-theme="dark"></div>
    </div>
  `;
  return {
    scope: document.querySelector<HTMLElement>("[data-agent-server-ui]")!,
    themeRoot: document.querySelector<HTMLElement>("[data-theme]")!,
  };
}

describe("color themes", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.style.removeProperty("color-scheme");
    document.getElementById("oh-color-theme-override")?.remove();
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("offers Light+ and Solarized Light in the theme selector", () => {
    expect(AVAILABLE_COLOR_THEMES).toEqual(
      expect.arrayContaining([
        { key: "light-plus", label: "Light+" },
        { key: "solarized-light", label: "Solarized Light" },
      ]),
    );
  });

  it("applies Light+ to the Canvas scope and HeroUI root", () => {
    const { scope, themeRoot } = renderThemeScope();

    applyColorTheme("light-plus");

    expect(scope.dataset.colorTheme).toBe("light-plus");
    expect(scope.dataset.colorScheme).toBe("light");
    expect(scope.style.colorScheme).toBe("light");
    expect(scope.style.getPropertyValue("--oh-background")).toBe("#FFFFFF");
    expect(scope.style.getPropertyValue("--oh-foreground")).toBe("#1F1F1F");
    expect(themeRoot.dataset.theme).toBe("light");
    expect(themeRoot).toHaveClass("light");
    expect(themeRoot).not.toHaveClass("dark");
    expect(document.documentElement.style.colorScheme).toBe("light");

    const css = document.getElementById("oh-color-theme-override")?.textContent;
    expect(css).toContain("color-scheme: light");
    expect(css).toContain("--cool-grey-950: #FFFFFF");
    expect(css).toContain("--heroui-background: 0 0% 100%");
  });

  it("applies Solarized Light and persists the preference", () => {
    const { scope } = renderThemeScope();

    setColorTheme("solarized-light");

    expect(readPersistedColorTheme()).toBe("solarized-light");
    expect(scope.style.getPropertyValue("--oh-background")).toBe("#FDF6E3");
    expect(scope.style.getPropertyValue("--oh-accent")).toBe("#268BD2");
  });

  it("restores dark appearance and clears light-only semantic overrides", () => {
    const { scope, themeRoot } = renderThemeScope();
    applyColorTheme("light-plus");

    applyColorTheme("openhands-neutral");

    expect(scope.dataset.colorScheme).toBe("dark");
    expect(scope.style.getPropertyValue("--oh-background")).toBe("");
    expect(themeRoot.dataset.theme).toBe("dark");
    expect(themeRoot).toHaveClass("dark");
    expect(themeRoot).not.toHaveClass("light");
  });

  it("notifies reactive consumers after the new theme is fully applied", () => {
    renderThemeScope();
    const listener = vi.fn();
    const unsubscribe = subscribeColorTheme(listener);

    applyColorTheme("solarized-light");

    expect(listener).toHaveBeenCalledOnce();
    unsubscribe();
  });
});
