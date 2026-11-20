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
      .find(
        (button) => button.getAttribute("aria-label") !== "Show suggestions",
      );
    if (!clearButton) {
      throw new Error("Clear button not found");
    }
    await user.click(clearButton);

    expect(screen.getByLabelText("Language")).toHaveValue("");
    expect(onSelectionChange).toHaveBeenCalledWith(null);
    expect(onInputChange).toHaveBeenCalledWith("");
  });

  it("selects an option with the keyboard", async () => {
    // Arrange
    const user = userEvent.setup();
    const onSelectionChange = vi.fn();

    render(
      <SettingsDropdownInput
        items={items}
        label="Language"
        name="language"
        onSelectionChange={onSelectionChange}
        testId="language-input"
      />,
    );

    // Act: open with the keyboard, move down and commit with Enter.
    const input = screen.getByLabelText("Language");
    input.focus();
    await user.keyboard("{ArrowDown}{ArrowDown}{Enter}");

    // Assert
    expect(onSelectionChange).toHaveBeenCalledWith("spanish");
    expect(input).toHaveValue("Spanish");
  });

  it("keeps a custom value on blur when allowsCustomValue is set", async () => {
    // Arrange
    const user = userEvent.setup();
    const onSelectionChange = vi.fn();
    const onInputChange = vi.fn();

    render(
      <div>
        <SettingsDropdownInput
          allowsCustomValue
          items={items}
          label="Language"
          name="language"
          onInputChange={onInputChange}
          onSelectionChange={onSelectionChange}
          testId="language-input"
        />
        <button type="button">Outside</button>
      </div>,
    );

    // Act
    const input = screen.getByLabelText("Language");
    await user.click(input);
    await user.keyboard("Klingon");
    await user.click(screen.getByText("Outside"));

    // Assert: the typed value survives blur and is never forced into a
    // selection change; the parent owns it through onInputChange.
    expect(input).toHaveValue("Klingon");
    expect(onInputChange).toHaveBeenLastCalledWith("Klingon");
    expect(onSelectionChange).not.toHaveBeenCalledWith("Klingon");
  });

  it("discards a non-matching value on blur without allowsCustomValue", async () => {
    // Arrange
    const user = userEvent.setup();
    const onSelectionChange = vi.fn();

    render(
      <div>
        <SettingsDropdownInput
          defaultSelectedKey="english"
          items={items}
          label="Language"
          name="language"
          onSelectionChange={onSelectionChange}
          testId="language-input"
        />
        <button type="button">Outside</button>
      </div>,
    );

    // Act: replace the selection with a partial, non-matching value, then blur.
    const input = screen.getByLabelText("Language");
    await user.click(input);
    await user.clear(input);
    await user.keyboard("Germ");
    await user.click(screen.getByText("Outside"));

    // Assert
    expect(input).toHaveValue("");
    expect(onSelectionChange).toHaveBeenLastCalledWith(null);
  });

  it("clears the selection when the input is emptied and blurred", async () => {
    // Arrange
    const user = userEvent.setup();
    const onSelectionChange = vi.fn();
    const onInputChange = vi.fn();

    render(
      <div>
        <SettingsDropdownInput
          defaultSelectedKey="english"
          items={items}
          label="Language"
          name="language"
          onInputChange={onInputChange}
          onSelectionChange={onSelectionChange}
          testId="language-input"
        />
        <button type="button">Outside</button>
      </div>,
    );

    // Act
    const input = screen.getByLabelText("Language");
    await user.click(input);
    await user.clear(input);
    await user.click(screen.getByText("Outside"));

    // Assert
    expect(input).toHaveValue("");
    expect(onSelectionChange).toHaveBeenCalledWith(null);
    expect(onInputChange).toHaveBeenLastCalledWith("");
  });

  it("applies a custom defaultFilter to the options", async () => {
    // Arrange: the filter matches a suffix, unlike the default
    // contains-filter, so "man" only keeps the label ending in it.
    const user = userEvent.setup();

    render(
      <SettingsDropdownInput
        defaultFilter={(textValue, inputValue) =>
          textValue.toLowerCase().endsWith(inputValue.toLowerCase())
        }
        items={items}
        label="Language"
        name="language"
        testId="language-input"
      />,
    );

    // Act
    const input = screen.getByLabelText("Language");
    await user.click(input);
    await user.keyboard("man");

    // Assert
    expect(await screen.findByText("German")).toBeInTheDocument();
    expect(screen.queryByText("English")).not.toBeInTheDocument();
    expect(screen.queryByText("Spanish")).not.toBeInTheDocument();
  });

  it("disables the clear button when the input is disabled", () => {
    // Arrange / Act
    render(
      <SettingsDropdownInput
        defaultSelectedKey="english"
        isClearable
        isDisabled
        items={items}
        label="Language"
        name="language"
        testId="language-input"
      />,
    );

    // Assert
    const clearButton = screen
      .getAllByRole("button")
      .find(
        (button) => button.getAttribute("aria-label") !== "Show suggestions",
      );
    expect(clearButton).toBeDefined();
    expect(clearButton).toBeDisabled();
    expect(screen.getByLabelText("Language")).toBeDisabled();
  });

  it("renders a single field border: shell carries the field style, input is borderless", () => {
    // Arrange / Act
    render(
      <SettingsDropdownInput
        items={items}
        label="Language"
        name="language"
        testId="language-input"
      />,
    );

    // Assert
    const input = screen.getByLabelText("Language");
    expect(input.className).toContain("border-0");
    expect(input.className).toContain("bg-transparent");
    expect(input.className).toContain("focus:ring-0");
    const group = input.closest('[data-slot="combo-box-input-group"]');
    expect(group?.className).toContain("border");
    expect(group?.className).toContain("border-control");
    expect(group?.className).toContain("focus-within:ring-1");
  });
});
