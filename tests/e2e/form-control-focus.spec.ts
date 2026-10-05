import { expect, test } from "@playwright/test";

for (const theme of [
  "openhands-neutral",
  "light-plus",
  "solarized-light",
] as const) {
  test(`${theme} preserves shared control borders through focus and blur`, async ({
    page,
    browserName,
  }) => {
    // macOS WebKit uses Option+Tab to include buttons in keyboard traversal.
    const nextControlKey =
      browserName === "webkit" && process.platform === "darwin"
        ? "Alt+Tab"
        : "Tab";
    await page.addInitScript((colorTheme) => {
      localStorage.setItem("analytics-consent", "false");
      localStorage.setItem("openhands-telemetry-consent", "denied");
      localStorage.setItem("openhands-telemetry-first-use", "true");
      localStorage.setItem("openhands-onboarded", "1");
      localStorage.setItem("openhands-color-theme", colorTheme);
      localStorage.setItem(
        "openhands-backends",
        JSON.stringify([
          {
            id: "default-local",
            name: "Local",
            host: window.location.origin,
            apiKey: "",
            kind: "local",
          },
        ]),
      );
      localStorage.setItem(
        "openhands-active-backend",
        JSON.stringify({ backendId: "default-local", orgId: null }),
      );
    }, theme);
    await page.route(/posthog|z\.openhands\.dev/, (route) => route.abort());
    await page.goto("/settings/app");
    const field = page.getByTestId("git-user-name-input");
    await expect(field).toBeVisible();
    // Local mock settings may still have null server-side consent even when
    // browser storage is seeded. Dismiss that modal without enabling tracking.
    const consent = page.getByTestId("telemetry-consent-form");
    const hasConsentPrompt = await consent
      .waitFor({ state: "visible", timeout: 5_000 })
      .then(
        () => true,
        () => false,
      );
    if (hasConsentPrompt) {
      await consent.getByRole("checkbox").uncheck();
      await consent.getByTestId("confirm-telemetry-preferences").click();
      await expect(consent).toBeHidden();
    }
    await expect(page.locator("[data-color-theme]").first()).toHaveAttribute(
      "data-color-theme",
      theme,
    );

    // Exercise the actual SettingsInput plus minimal consumers of the shared
    // shell/trigger helpers under the app's real compiled CSS and theme scope.
    // Vite serves these production exports; no utility strings or CSS rules
    // are copied into the fixture, so removing a dark focus override regresses
    // the computed border rather than merely failing a string assertion.
    const expected = await page.evaluate(async (colorTheme) => {
      const formModule = "/src/utils/form-control-classes.ts";
      const dropdownModule = "/src/utils/dropdown-classes.ts";
      const { formControlShellClassName, formControlInlineInputClassName } =
        await import(/* @vite-ignore */ formModule);
      const { dropdownFilterTriggerClassName } = await import(
        /* @vite-ignore */ dropdownModule
      );
      const scope = document.querySelector(
        "[data-agent-server-ui] [data-theme]",
      );
      if (!scope) throw new Error("The themed app scope was not mounted");
      const fixture = document.createElement("section");
      fixture.dataset.testid = "focus-cascade-fixture";
      fixture.style.cssText =
        "position:fixed;bottom:16px;right:16px;z-index:9999;display:flex;gap:12px;padding:12px;background:var(--oh-background)";
      const shell = document.createElement("div");
      shell.dataset.testid = "focus-cascade-shell";
      shell.className = formControlShellClassName;
      const input = document.createElement("input");
      input.setAttribute("aria-label", "Focus cascade search");
      input.className = formControlInlineInputClassName;
      shell.append(input);
      const trigger = document.createElement("button");
      trigger.type = "button";
      trigger.dataset.testid = "focus-cascade-trigger";
      trigger.className = dropdownFilterTriggerClassName;
      trigger.textContent = "Focus cascade filter";
      const end = document.createElement("button");
      end.type = "button";
      end.textContent = "End focus check";
      const reference = document.createElement("span");
      reference.style.visibility = "hidden";
      fixture.append(shell, trigger, end, reference);
      scope.append(fixture);

      // Let this browser normalize color-mix(), avoiding differences in color
      // serialization between Chromium, Firefox, and WebKit.
      reference.style.borderColor =
        colorTheme === "openhands-neutral"
          ? "var(--oh-border)"
          : "var(--oh-border-input)";
      const restingBorder = getComputedStyle(reference).borderTopColor;
      reference.style.borderColor =
        "color-mix(in oklab, var(--oh-contrast) 40%, transparent)";
      reference.style.backgroundColor =
        "color-mix(in oklab, var(--oh-contrast) 20%, transparent)";
      return {
        restingBorder,
        focusedBorder: getComputedStyle(reference).borderTopColor,
        ringColor: getComputedStyle(reference).backgroundColor,
      };
    }, theme);

    const shell = page.getByTestId("focus-cascade-shell");
    const search = page.getByRole("textbox", { name: "Focus cascade search" });
    const trigger = page.getByTestId("focus-cascade-trigger");
    for (const control of [field, shell, trigger]) {
      await expect(control).toHaveCSS(
        "border-top-color",
        expected.restingBorder,
      );
    }

    await field.click();
    await expect(field).toHaveCSS("border-top-color", expected.focusedBorder);
    await expect
      .poll(() => field.evaluate((el) => getComputedStyle(el).boxShadow))
      .toContain(expected.ringColor);
    await field.press(nextControlKey);
    await expect(field).toHaveCSS("border-top-color", expected.restingBorder);

    // Tab from the nested input exercises :focus-within on the shell and real
    // keyboard :focus-visible on the trigger, including repeated entry/exit.
    for (let cycle = 0; cycle < 2; cycle += 1) {
      await search.click();
      await expect(shell).toHaveCSS("border-top-color", expected.focusedBorder);
      await search.press(nextControlKey);
      await expect(shell).toHaveCSS("border-top-color", expected.restingBorder);
      await expect(trigger).toBeFocused();
      await expect
        .poll(() => trigger.evaluate((el) => el.matches(":focus-visible")))
        .toBe(true);
      await expect(trigger).toHaveCSS(
        "border-top-color",
        expected.focusedBorder,
      );
      await expect
        .poll(() => trigger.evaluate((el) => getComputedStyle(el).boxShadow))
        .toContain(expected.ringColor);
      await trigger.press(nextControlKey);
      await expect(trigger).toHaveCSS(
        "border-top-color",
        expected.restingBorder,
      );
    }
  });
}
