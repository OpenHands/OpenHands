import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import { SettingsDropdownInput } from "./settings-dropdown-input";

describe("SettingsDropdownInput", () => {
  it("uses an explicit accessible name without rendering an extra label", () => {
    renderWithProviders(
      <SettingsDropdownInput
        testId="agent-settings-router-select"
        name="agent-router"
        label=""
        ariaLabel="Model Router"
        selectedKey="off"
        items={[{ key: "off", label: "Off" }]}
      />,
    );

    expect(screen.getByRole("combobox", { name: "Model Router" })).toHaveValue(
      "Off",
    );
    expect(screen.queryByText("Model Router")).not.toBeInTheDocument();
  });

  it("keeps the chevron trigger in flow after the clear button so they do not overlap", () => {
    renderWithProviders(
      <SettingsDropdownInput
        testId="title-llm-profile-input"
        name="title-llm-profile"
        ariaLabel="Title model"
        selectedKey="flash"
        items={[{ key: "flash", label: "deepseek-flash" }]}
      />,
    );

    const clear = screen.getByRole("button", {
      name: "COMMON$CLEAR_SELECTION",
    });
    const trigger = clear.nextElementSibling as HTMLElement;

    // HeroUI's `.combo-box__trigger` is `absolute end-0`, which put the
    // chevron on top of the clear button. jsdom has no layout, so assert the
    // overrides that keep both buttons in normal flow side by side.
    expect(trigger).toHaveClass(
      "combo-box__trigger",
      "static",
      "translate-y-0",
    );
    expect(trigger.parentElement).toHaveClass("pe-0");
  });
});
