import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { Dropdown } from "#/ui/dropdown/dropdown";

const mockOptions = [
  { value: "1", label: "Option 1" },
  { value: "2", label: "Option 2" },
  { value: "3", label: "Option 3" },
];

describe("Dropdown", () => {
  describe("Trigger", () => {
    it("should render a custom trigger button", () => {
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");

      expect(trigger).toBeInTheDocument();
    });

    it("should open dropdown on trigger click", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      expect(screen.queryByText("Option 1")).not.toBeInTheDocument();

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      const listbox = screen.getByRole("listbox");
      expect(listbox).toBeInTheDocument();
      expect(screen.getByText("Option 1")).toBeInTheDocument();
      expect(screen.getByText("Option 2")).toBeInTheDocument();
      expect(screen.getByText("Option 3")).toBeInTheDocument();
    });
  });

  describe("Hidden trigger", () => {
    it("keeps Downshift controls connected while selecting from the popover", async () => {
      // Arrange: collapsed navigation supplies its own visible trigger.
      const user = userEvent.setup();
      const onChange = vi.fn();
      const { container } = render(
        <Dropdown
          options={mockOptions}
          placeholder="Choose a backend"
          hideTrigger
          defaultOpen
          onChange={onChange}
        />,
      );
      const input = screen.getByRole("combobox", {
        name: "Choose a backend",
      });
      const toggle = container.querySelector("button[aria-hidden]");
      expect(input).toHaveAttribute("tabindex", "-1");
      expect(input).toHaveAttribute(
        "aria-controls",
        screen.getByRole("listbox").id,
      );
      expect(toggle).toHaveAttribute("tabindex", "-1");
      expect(toggle).toHaveAttribute("aria-hidden", "true");
      expect(screen.queryByTestId("dropdown-trigger")).not.toBeInTheDocument();

      // Act: choose through the visible option, not the hidden input or toggle.
      await user.click(screen.getByRole("option", { name: "Option 2" }));

      // Assert: the same real combobox owns the selected value and closes.
      expect(onChange).toHaveBeenCalledExactlyOnceWith(mockOptions[1]);
      expect(input).toHaveValue("Option 2");
      expect(input).toHaveAttribute("aria-expanded", "false");
    });
  });

  describe("Type-ahead / Search", () => {
    it("should filter options based on input text", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      const input = screen.getByRole("combobox");
      await user.type(input, "Option 1");

      expect(screen.getByText("Option 1")).toBeInTheDocument();
      expect(screen.queryByText("Option 2")).not.toBeInTheDocument();
      expect(screen.queryByText("Option 3")).not.toBeInTheDocument();
    });

    it("should be case-insensitive by default", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      const input = screen.getByRole("combobox");
      await user.type(input, "option 1");

      expect(screen.getByText("Option 1")).toBeInTheDocument();
      expect(screen.queryByText("Option 2")).not.toBeInTheDocument();
    });

    it("should show all options when search is cleared", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      const input = screen.getByRole("combobox");
      await user.type(input, "Option 1");
      await user.clear(input);

      expect(screen.getByText("Option 1")).toBeInTheDocument();
      expect(screen.getByText("Option 2")).toBeInTheDocument();
      expect(screen.getByText("Option 3")).toBeInTheDocument();
    });
  });

  describe("Empty state", () => {
    it("should display empty state when no options provided", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={[]} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      expect(screen.getByText("No options")).toBeInTheDocument();
    });

    it("should render custom empty state message", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={[]} emptyMessage="Nothing found" />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      expect(screen.getByText("Nothing found")).toBeInTheDocument();
    });
  });

  describe("Single selection", () => {
    it("should select an option on click", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      const option = screen.getByText("Option 1");
      await user.click(option);

      expect(screen.getByRole("combobox")).toHaveValue("Option 1");
    });

    it("should close dropdown after selection", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      await user.click(screen.getByText("Option 1"));

      expect(screen.queryByText("Option 2")).not.toBeInTheDocument();
    });

    it("should display selected option in input", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      await user.click(screen.getByText("Option 1"));

      expect(screen.getByRole("combobox")).toHaveValue("Option 1");
    });

    it("should highlight currently selected option in list", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      await user.click(screen.getByRole("option", { name: "Option 1" }));

      await user.click(trigger);

      const selectedOption = screen.getByRole("option", { name: "Option 1" });
      expect(selectedOption).toHaveAttribute("aria-selected", "true");
    });

    it("should clear the search input and show all options when reopening dropdown", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      await user.click(screen.getByRole("option", { name: "Option 1" }));

      // Reopen the dropdown
      await user.click(trigger);

      const input = screen.getByRole("combobox");
      expect(input).toHaveValue("");
      expect(
        screen.getByRole("option", { name: "Option 1" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "Option 2" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "Option 3" }),
      ).toBeInTheDocument();
    });
  });

  describe("Full label on hover", () => {
    it("should expose the full label as a title on the trigger and each option", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} defaultValue={mockOptions[1]} />);

      expect(screen.getByRole("combobox")).toHaveAttribute("title", "Option 2");

      await user.click(screen.getByTestId("dropdown-trigger"));

      mockOptions.forEach(({ label }) => {
        expect(screen.getByText(label)).toHaveAttribute("title", label);
      });
    });
  });

  describe("Clear button", () => {
    it("should not render clear button by default", () => {
      render(<Dropdown options={mockOptions} />);

      expect(screen.queryByTestId("dropdown-clear")).not.toBeInTheDocument();
    });

    it("should render clear button when clearable prop is true and has value", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} clearable />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      await user.click(screen.getByRole("option", { name: "Option 1" }));

      expect(screen.getByTestId("dropdown-clear")).toBeInTheDocument();
    });

    it("should clear selection and search input when clear button is clicked", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} clearable />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      await user.click(screen.getByRole("option", { name: "Option 1" }));

      const clearButton = screen.getByTestId("dropdown-clear");
      await user.click(clearButton);

      expect(screen.getByRole("combobox")).toHaveValue("");
    });

    it("should not render clear button when there is no selection", () => {
      render(<Dropdown options={mockOptions} clearable />);

      expect(screen.queryByTestId("dropdown-clear")).not.toBeInTheDocument();
    });

    it("should show placeholder after clearing selection", async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          options={mockOptions}
          clearable
          placeholder="Select an option"
        />,
      );

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      await user.click(screen.getByRole("option", { name: "Option 1" }));

      const clearButton = screen.getByTestId("dropdown-clear");
      await user.click(clearButton);

      const input = screen.getByRole("combobox");
      expect(input).toHaveValue("");
    });
  });

  describe("Loading state", () => {
    it("should not display loading indicator by default", () => {
      render(<Dropdown options={mockOptions} />);

      expect(screen.queryByTestId("dropdown-loading")).not.toBeInTheDocument();
    });

    it("should display loading indicator when loading prop is true", () => {
      render(<Dropdown options={mockOptions} loading />);

      expect(screen.getByTestId("dropdown-loading")).toBeInTheDocument();
    });

    it("should disable interaction while loading", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} loading />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      expect(trigger).toHaveAttribute("aria-expanded", "false");
    });
  });

  describe("Disabled state", () => {
    it("should not open dropdown when disabled", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} disabled />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      expect(trigger).toHaveAttribute("aria-expanded", "false");
    });

    it("should have disabled attribute on trigger", () => {
      render(<Dropdown options={mockOptions} disabled />);

      const trigger = screen.getByTestId("dropdown-trigger");
      expect(trigger).toBeDisabled();
    });
  });

  describe("Placeholder", () => {
    it("should display placeholder text when no value selected", () => {
      render(<Dropdown options={mockOptions} placeholder="Select an option" />);

      const input = screen.getByRole("combobox");
      expect(input).toHaveAttribute("placeholder", "Select an option");
    });
  });

  describe("Default value", () => {
    it("should display defaultValue in input on mount", () => {
      render(<Dropdown options={mockOptions} defaultValue={mockOptions[0]} />);

      const input = screen.getByRole("combobox");
      expect(input).toHaveValue("Option 1");
    });

    it("should show all options when opened with defaultValue", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} defaultValue={mockOptions[0]} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      expect(
        screen.getByRole("option", { name: "Option 1" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "Option 2" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "Option 3" }),
      ).toBeInTheDocument();
    });

    it("should restore input value when closed with Escape", async () => {
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} defaultValue={mockOptions[0]} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      const input = screen.getByRole("combobox");
      await user.type(input, "test");
      await user.keyboard("{Escape}");

      expect(input).toHaveValue("Option 1");
    });
  });

  describe("onChange", () => {
    it("should call onChange with selected item when option is clicked", async () => {
      const user = userEvent.setup();
      const onChangeMock = vi.fn();
      render(<Dropdown options={mockOptions} onChange={onChangeMock} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);
      await user.click(screen.getByRole("option", { name: "Option 1" }));

      expect(onChangeMock).toHaveBeenCalledWith(mockOptions[0]);
    });

    it("should call onChange with null when selection is cleared", async () => {
      const user = userEvent.setup();
      const onChangeMock = vi.fn();
      render(
        <Dropdown
          options={mockOptions}
          clearable
          defaultValue={mockOptions[0]}
          onChange={onChangeMock}
        />,
      );

      const clearButton = screen.getByTestId("dropdown-clear");
      await user.click(clearButton);

      expect(onChangeMock).toHaveBeenCalledWith(null);
    });
  });

  describe("onOpenChange", () => {
    it("reports when the menu opens and closes", async () => {
      const user = userEvent.setup();
      const onOpenChange = vi.fn();
      render(<Dropdown options={mockOptions} onOpenChange={onOpenChange} />);

      await user.click(screen.getByTestId("dropdown-trigger"));
      expect(onOpenChange).toHaveBeenLastCalledWith(true);

      await user.click(screen.getByRole("option", { name: "Option 2" }));
      expect(onOpenChange).toHaveBeenLastCalledWith(false);
    });
  });

  describe("Menu anchoring", () => {
    it("anchors the menu to the trigger by default and to the parent row when asked", () => {
      // The absolutely positioned menu takes its width and position from the
      // nearest positioned ancestor: the trigger wrapper unless anchored to the parent.
      const { rerender } = render(
        <Dropdown options={mockOptions} testId="picker" />,
      );
      expect(screen.getByTestId("picker")).toHaveClass("relative");

      rerender(
        <Dropdown options={mockOptions} testId="picker" anchorMenuToParent />,
      );
      expect(screen.getByTestId("picker")).not.toHaveClass("relative");
    });
  });

  describe("Grouped options", () => {
    it("lists consecutive options under one heading with their short labels", async () => {
      const user = userEvent.setup();
      const onChange = vi.fn();
      const grouped = [
        { value: "local", label: "Local" },
        {
          value: "a",
          label: "Cloud – Alpha",
          group: "Cloud",
          groupLabel: "Alpha",
        },
        {
          value: "b",
          label: "Cloud – Beta",
          group: "Cloud",
          groupLabel: "Beta",
        },
      ];
      render(<Dropdown options={grouped} onChange={onChange} />);

      await user.click(screen.getByTestId("dropdown-trigger"));

      expect(screen.getAllByText("Cloud")).toHaveLength(1);
      // Assistive tech still hears the group with each row.
      const beta = screen.getByRole("option", { name: "Cloud – Beta" });
      expect(beta).toHaveTextContent(/^Beta$/);
      expect(within(beta).getByTitle("Cloud – Beta")).toBeInTheDocument();
      await user.click(beta);
      expect(onChange).toHaveBeenCalledWith(grouped[2]);
    });
  });

  describe("Controlled mode", () => {
    it.todo("should reflect external value changes");
    it.todo("should call onChange when selection changes");
    it.todo("should not update internal state when controlled");
  });

  describe("Uncontrolled mode", () => {
    it.todo("should manage selection state internally");
    it.todo("should call onChange when selection changes");
    it.todo("should support defaultValue prop");
  });

  describe("testId prop", () => {
    it("should apply custom testId to the root container", () => {
      render(<Dropdown options={mockOptions} testId="org-dropdown" />);

      expect(screen.getByTestId("org-dropdown")).toBeInTheDocument();
    });
  });

  describe("Cursor position preservation", () => {
    it("should keep menu open when clicking the input while dropdown is open", async () => {
      // Without a stateReducer, Downshift's default InputClick behavior
      // toggles the menu (closes it if already open). The stateReducer
      // should override this to keep the menu open so users can click
      // to reposition their cursor without losing the dropdown.
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      // Menu should be open
      expect(screen.getByText("Option 1")).toBeInTheDocument();

      // Click the input itself (simulates clicking to reposition cursor)
      const input = screen.getByRole("combobox");
      await user.click(input);

      // Menu should still be open — not toggled closed
      expect(screen.getByText("Option 1")).toBeInTheDocument();
    });

    it("should still filter options correctly after typing with cursor fix", async () => {
      // Verifies that the direct onChange handler (which bypasses Downshift's
      // default onInputValueChange for cursor preservation) still updates
      // the search/filter state correctly.
      const user = userEvent.setup();
      render(<Dropdown options={mockOptions} />);

      const trigger = screen.getByTestId("dropdown-trigger");
      await user.click(trigger);

      const input = screen.getByRole("combobox");
      await user.type(input, "Option 1");

      expect(screen.getByText("Option 1")).toBeInTheDocument();
      expect(screen.queryByText("Option 2")).not.toBeInTheDocument();
      expect(screen.queryByText("Option 3")).not.toBeInTheDocument();
    });
  });
});
