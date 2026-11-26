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
});
