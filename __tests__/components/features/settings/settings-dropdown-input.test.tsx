import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

const items = [
  { key: "english", label: "English" },
  { key: "spanish", label: "Spanish" },
  { key: "german", label: "German" },
];

describe("SettingsDropdownInput", () => {
  it("shows the default selection and emits selection changes", async () => {
    const user = userEvent.setup();
    const onSelectionChange = vi.fn();

    render(
      <SettingsDropdownInput
        defaultSelectedKey="english"
        items={items}
        label="Language"
        name="language"
        onSelectionChange={onSelectionChange}
        testId="language-input"
      />,
    );

    const input = screen.getByLabelText("Language");

    expect(input).toHaveValue("English");

    await user.click(input);
    await user.click(await screen.findByText("Spanish"));

    expect(onSelectionChange).toHaveBeenCalledWith("spanish");
  });

  it("filters options while typing", async () => {
    const user = userEvent.setup();

    render(
      <SettingsDropdownInput
        defaultSelectedKey="english"
        items={items}
        label="Language"
        name="language"
        testId="language-input"
      />,
    );

    const input = screen.getByLabelText("Language");
    await user.click(input);
    await user.clear(input);
    await user.keyboard("Germ");

    expect(await screen.findByText("German")).toBeInTheDocument();
    expect(screen.queryByText("English")).not.toBeInTheDocument();
    expect(screen.queryByText("Spanish")).not.toBeInTheDocument();
  });

  it("clears the selection when the clear button is pressed", async () => {
    const user = userEvent.setup();
    const onSelectionChange = vi.fn();
    const onInputChange = vi.fn();

    render(
      <SettingsDropdownInput
        defaultSelectedKey="english"
        isClearable
        items={items}
        label="Language"
        name="language"
        onInputChange={onInputChange}
        onSelectionChange={onSelectionChange}
        testId="language-input"
      />,
    );

    const clearButton = screen
      .getAllByRole("button")
      .find((button) => button.getAttribute("aria-label") !== "Show suggestions");
    if (!clearButton) {
      throw new Error("Clear button not found");
    }
    await user.click(clearButton);

    expect(screen.getByLabelText("Language")).toHaveValue("");
    expect(onSelectionChange).toHaveBeenCalledWith(null);
    expect(onInputChange).toHaveBeenCalledWith("");
  });
});
