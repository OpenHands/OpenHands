import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RecommendedAutomationsSwitch } from "#/components/features/settings/app-settings/recommended-automations-switch";
import { HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_STORAGE_KEY } from "#/components/features/home/home-recommended-automations.constants";
import { readHomeRecommendedAutomationsDismissed } from "#/components/features/home/home-recommended-automations-storage";

describe("RecommendedAutomationsSwitch", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  // @spec HL-002 — The switch defaults to on and writes the shared preference.
  it("shows recommended automations by default and hides them when toggled off", async () => {
    const user = userEvent.setup();
    render(<RecommendedAutomationsSwitch />);

    const toggle = screen.getByTestId("show-recommended-automations-switch");
    expect(toggle).toBeChecked();
    expect(readHomeRecommendedAutomationsDismissed()).toBe(false);

    await user.click(toggle);

    expect(toggle).not.toBeChecked();
    expect(
      window.localStorage.getItem(
        HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_STORAGE_KEY,
      ),
    ).toBe("true");
  });

  // @spec HL-002 — Turning the switch back on clears the dismissal.
  it("re-enables recommended automations when toggled back on", async () => {
    window.localStorage.setItem(
      HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_STORAGE_KEY,
      "true",
    );

    const user = userEvent.setup();
    render(<RecommendedAutomationsSwitch />);

    const toggle = screen.getByTestId("show-recommended-automations-switch");
    expect(toggle).not.toBeChecked();

    await user.click(toggle);

    expect(toggle).toBeChecked();
    expect(readHomeRecommendedAutomationsDismissed()).toBe(false);
  });
});
