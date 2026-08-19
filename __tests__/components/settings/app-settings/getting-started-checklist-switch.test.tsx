import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GettingStartedChecklistSwitch } from "#/components/features/settings/app-settings/getting-started-checklist-switch";
import { SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_STORAGE_KEY } from "#/components/features/sidebar/sidebar-onboarding-checklist.constants";
import { readSidebarOnboardingChecklistDismissed } from "#/components/features/sidebar/sidebar-onboarding-checklist-storage";

describe("GettingStartedChecklistSwitch", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("shows the checklist by default and hides it when toggled off", async () => {
    const user = userEvent.setup();
    render(<GettingStartedChecklistSwitch />);

    const toggle = screen.getByTestId("show-getting-started-checklist-switch");
    expect(toggle).toBeChecked();
    expect(readSidebarOnboardingChecklistDismissed()).toBe(false);

    await user.click(toggle);

    expect(toggle).not.toBeChecked();
    expect(
      window.localStorage.getItem(
        SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_STORAGE_KEY,
      ),
    ).toBe("true");
  });

  it("re-enables the checklist when toggled back on", async () => {
    window.localStorage.setItem(
      SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_STORAGE_KEY,
      "true",
    );

    const user = userEvent.setup();
    render(<GettingStartedChecklistSwitch />);

    const toggle = screen.getByTestId("show-getting-started-checklist-switch");
    expect(toggle).not.toBeChecked();

    await user.click(toggle);

    expect(toggle).toBeChecked();
    expect(readSidebarOnboardingChecklistDismissed()).toBe(false);
  });
});
