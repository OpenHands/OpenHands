import { expect, test, type Locator } from "@playwright/test";

async function renderedContrast(
  locator: Locator,
  foregroundProperty: "color" | "stroke" = "color",
): Promise<number> {
  return locator.evaluate((el, property) => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 1;
    const context = canvas.getContext("2d")!;
    const luminance = (color: string) => {
      context.clearRect(0, 0, 1, 1);
      context.fillStyle = color;
      context.fillRect(0, 0, 1, 1);
      const [r, g, b] = context.getImageData(0, 0, 1, 1).data;
      const channels = [r, g, b].map((value) => {
        const c = value / 255;
        return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
      });
      return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
    };
    let surface: Element | null = el;
    let background = "rgba(0, 0, 0, 0)";
    while (surface && background === "rgba(0, 0, 0, 0)") {
      background = getComputedStyle(surface).backgroundColor;
      surface = surface.parentElement;
    }
    const ink = luminance(getComputedStyle(el)[property]);
    const fill = luminance(background);
    return (Math.max(ink, fill) + 0.05) / (Math.min(ink, fill) + 0.05);
  }, foregroundProperty);
}

for (const theme of [
  "openhands-neutral",
  "openhands-deepsea",
  "openhands-neo",
  "light-plus",
  "solarized-light",
  "paper",
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
      const automationModule =
        "/src/components/features/automations/automation-view-mode.ts";
      const { automationActivityRowClassName } = await import(
        /* @vite-ignore */ automationModule
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
      const row = document.createElement("div");
      row.dataset.testid = "focus-cascade-automation-row";
      row.className = automationActivityRowClassName;
      const rowButton = document.createElement("button");
      rowButton.textContent = "Open automation";
      row.append(rowButton);
      const end = document.createElement("button");
      end.type = "button";
      end.textContent = "End focus check";
      const reference = document.createElement("span");
      reference.style.visibility = "hidden";
      fixture.append(shell, trigger, row, end, reference);
      scope.append(fixture);

      // Let this browser normalize color-mix(), avoiding differences in color
      // serialization between Chromium, Firefox, and WebKit.
      reference.style.borderColor = ![
        "light-plus",
        "solarized-light",
        "paper",
      ].includes(colorTheme)
        ? "var(--oh-border)"
        : "var(--oh-border-input)";
      const restingBorder = getComputedStyle(reference).borderTopColor;
      reference.style.borderColor =
        "color-mix(in oklab, var(--oh-contrast) 40%, transparent)";
      reference.style.backgroundColor =
        "color-mix(in oklab, var(--oh-contrast) 20%, transparent)";
      const ringColor = getComputedStyle(reference).backgroundColor;
      reference.style.backgroundColor =
        colorTheme === "light-plus" || colorTheme === "solarized-light"
          ? "var(--oh-interactive-hover-low)"
          : "var(--oh-surface-raised)";
      return {
        restingBorder,
        focusedBorder: getComputedStyle(reference).borderTopColor,
        ringColor,
        rowHighlight: getComputedStyle(reference).backgroundColor,
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

    // Shared automation rows retain each palette's hover and keyboard cue.
    const row = page.getByTestId("focus-cascade-automation-row");
    // Embedded Canvas has one scoped root. Remove the SPA body's extra
    // scope marker so it cannot accidentally make a broken selector match.
    await row.evaluate((el) => {
      const root = el.closest("[data-agent-server-ui]");
      for (
        let ancestor = root?.parentElement;
        ancestor;
        ancestor = ancestor.parentElement
      ) {
        if (ancestor.hasAttribute("data-agent-server-ui")) {
          ancestor.setAttribute("data-restore-ui-scope", "");
          ancestor.removeAttribute("data-agent-server-ui");
        }
      }
    });
    await row.hover();
    await expect(row).toHaveCSS("background-color", expected.rowHighlight);
    const end = page.getByRole("button", { name: "End focus check" });
    await end.click();
    await end.press(
      browserName === "webkit" && process.platform === "darwin"
        ? "Alt+Shift+Tab"
        : "Shift+Tab",
    );
    await expect(row.getByRole("button")).toBeFocused();
    await expect(row).toHaveCSS("background-color", expected.rowHighlight);

    await page.evaluate(() => {
      for (const ancestor of document.querySelectorAll(
        "[data-restore-ui-scope]",
      )) {
        ancestor.setAttribute("data-agent-server-ui", "");
        ancestor.removeAttribute("data-restore-ui-scope");
      }
    });

    await page
      .getByTestId("focus-cascade-fixture")
      .evaluate((el) => el.remove());
    const fixtureModule = "/tests/e2e/support/theme-consumer-fixture.tsx";
    const appearanceOverride =
      theme === "openhands-neutral"
        ? "light"
        : theme === "light-plus"
          ? "dark"
          : undefined;
    await page.evaluate(
      async ({ modulePath, appearanceOverride }) => {
        const { mountThemeConsumerFixture } = await import(
          /* @vite-ignore */ modulePath
        );
        const scope = document.querySelector(
          "[data-agent-server-ui] [data-theme]",
        );
        if (!scope) throw new Error("The themed app scope was not mounted");
        const cleanup = mountThemeConsumerFixture(scope, appearanceOverride);
        (
          window as typeof window & { cleanupThemeFixture?: () => void }
        ).cleanupThemeFixture = cleanup;
      },
      { modulePath: fixtureModule, appearanceOverride },
    );
    const fixture = page.getByTestId("theme-consumer-fixture");
    if (appearanceOverride) {
      const embeddedField = page.getByTestId("theme-appearance-override");
      const expectedBorder = await embeddedField.evaluate((el, appearance) => {
        const reference = document.createElement("span");
        reference.style.borderColor =
          appearance === "light"
            ? "var(--oh-border-input)"
            : "var(--oh-border)";
        el.parentElement!.append(reference);
        const border = getComputedStyle(reference).borderTopColor;
        reference.remove();
        return border;
      }, appearanceOverride);
      await expect(embeddedField).toHaveCSS("border-top-color", expectedBorder);
    }
    const draft = fixture.getByTestId("active-status-badge-draft");
    await expect(draft).toBeVisible();
    if (["light-plus", "solarized-light", "paper"].includes(theme)) {
      expect(await renderedContrast(draft)).toBeGreaterThanOrEqual(4.5);
    }
    if (theme === "light-plus") {
      const extension = fixture
        .getByTestId("theme-uploaded-file")
        .getByText("TXT", { exact: true });
      expect(await renderedContrast(extension)).toBeGreaterThanOrEqual(4.5);
    }
    if (["light-plus", "solarized-light", "paper"].includes(theme)) {
      const toggle = fixture.getByTestId("theme-selected-toggle");
      await toggle.hover();
      await expect(toggle).toHaveAttribute("data-showing-remove", "true");
      const removeIcon = toggle.locator("svg path");
      await expect(removeIcon).toBeVisible();
      const hoverFill = await toggle.evaluate((el) => {
        const reference = document.createElement("span");
        reference.style.backgroundColor = "var(--oh-danger)";
        el.append(reference);
        const fill = getComputedStyle(reference).backgroundColor;
        reference.remove();
        return fill;
      });
      await expect(toggle).toHaveCSS("background-color", hoverFill);
      await expect
        .poll(() => renderedContrast(removeIcon, "stroke"))
        .toBeGreaterThanOrEqual(3);
    }
    const stateBorders = await fixture.evaluate((el) => {
      const reference = document.createElement("span");
      el.append(reference);
      reference.style.borderColor = "var(--oh-danger)";
      const invalid = getComputedStyle(reference).borderTopColor;
      reference.style.borderColor = "var(--oh-border-subtle)";
      const disabled = getComputedStyle(reference).borderTopColor;
      reference.remove();
      return { invalid, disabled };
    });
    await expect(fixture.getByTestId("theme-caller-invalid")).toHaveCSS(
      "border-top-color",
      stateBorders.invalid,
    );
    await expect(fixture.getByTestId("theme-error")).toHaveCSS(
      "border-top-color",
      stateBorders.invalid,
    );
    await expect(fixture.getByTestId("theme-error-error")).toHaveCSS(
      "color",
      stateBorders.invalid,
    );
    await expect(fixture.getByTestId("theme-disabled")).toHaveCSS(
      "border-top-color",
      stateBorders.disabled,
    );
    for (const override of [null, "rgb(97, 43, 151)"]) {
      const expectedInk = await fixture.evaluate((el, color) => {
        if (color)
          (el as HTMLElement).style.setProperty("--oh-contrast", color);
        const reference = document.createElement("span");
        reference.style.color = "var(--oh-contrast)";
        el.append(reference);
        const ink = getComputedStyle(reference).color;
        reference.remove();
        return ink;
      }, override);
      for (const selector of [
        '[data-testid="theme-branch-icon"] svg path',
        '[data-testid="theme-repo-icon"] svg path',
        '[data-testid="workspace-mode-selector"] svg path',
        ".git-external-link-icon svg path",
      ]) {
        await expect(fixture.locator(selector).first()).toHaveCSS(
          "fill",
          expectedInk,
        );
      }
    }
    await page.evaluate(() => {
      (
        window as typeof window & { cleanupThemeFixture?: () => void }
      ).cleanupThemeFixture?.();
    });
    await page.getByRole("link", { name: "Agent", exact: true }).click();
    await page.getByTestId("add-agent-profile").click();
    await page.getByTestId("agent-type-selector").click();
    await page
      .getByRole("option", { name: "ACP (external subprocess)", exact: true })
      .click();
    const command = page.getByTestId("agent-command-input");
    await expect(command).toBeVisible();
    const commandBorders = await command.evaluate((el) => {
      const reference = document.createElement("span");
      el.parentElement!.append(reference);
      reference.style.borderColor = "var(--oh-contrast)";
      const focused = getComputedStyle(reference).borderTopColor;
      const resting = getComputedStyle(el).borderTopColor;
      reference.remove();
      return { resting, focused };
    });
    expect(commandBorders.focused).not.toBe(commandBorders.resting);
    // Enter via the keyboard so this protects the textarea's actual focus
    // cascade, not just its class recipe or a synthetic CSS fixture.
    await command.click();
    // Native text fields participate in WebKit's plain Tab traversal too;
    // Option+Tab is only needed above to include the button trigger.
    await command.press("Shift+Tab");
    await page.keyboard.press("Tab");
    await expect(command).toBeFocused();
    await expect(command).toHaveCSS("border-top-color", commandBorders.focused);
    await command.press(nextControlKey);
    await expect(command).toHaveCSS("border-top-color", commandBorders.resting);
  });
}
