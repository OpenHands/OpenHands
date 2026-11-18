import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AddSkillModal } from "#/components/features/skills/add-skill-modal";
import { ADD_SKILL_EXAMPLE_COMMAND } from "#/constants/skills-docs";

const originalClipboard = navigator.clipboard;
const originalExecCommand = document.execCommand;
const originalSelect = HTMLTextAreaElement.prototype.select;

afterEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    writable: true,
    value: originalClipboard,
  });
  document.execCommand = originalExecCommand;
  HTMLTextAreaElement.prototype.select = originalSelect;
  vi.restoreAllMocks();
});

describe("AddSkillModal clipboard fallback", () => {
  it("keeps the temporary textarea focused inside the modal while copying", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      writable: true,
      value: undefined,
    });
    HTMLTextAreaElement.prototype.select = function select(
      this: HTMLTextAreaElement,
    ) {
      this.focus();
    };

    let activeDuringCopy: Element | null = null;
    let valueDuringCopy: string | null = null;
    document.execCommand = vi.fn(() => {
      const active = document.activeElement;
      activeDuringCopy = active;
      valueDuringCopy =
        active instanceof HTMLTextAreaElement ? active.value : null;
      return true;
    });

    render(<AddSkillModal onClose={vi.fn()} />);
    const copyButton = screen.getByTestId("add-skill-modal-example-copy");

    await user.click(copyButton);

    await waitFor(() =>
      expect(document.execCommand).toHaveBeenCalledWith("copy"),
    );
    expect(activeDuringCopy).toBeInstanceOf(HTMLTextAreaElement);
    expect(valueDuringCopy).toBe(ADD_SKILL_EXAMPLE_COMMAND);
    await waitFor(() => expect(copyButton).toBeDisabled());
  });
});
