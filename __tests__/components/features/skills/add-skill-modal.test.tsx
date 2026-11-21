import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { AddSkillModal } from "#/components/features/skills/add-skill-modal";
import { ADD_SKILL_EXAMPLE_COMMAND } from "#/constants/skills-docs";
import SkillsService from "#/api/skills-service";
import type { SkillInfo } from "#/types/settings";

const originalClipboard = navigator.clipboard;
const originalExecCommand = document.execCommand;
const originalSelect = HTMLTextAreaElement.prototype.select;

beforeEach(() => {
  localStorage.clear();
});

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

    renderWithProviders(<AddSkillModal onClose={vi.fn()} />);

    // Switch to "Import from repository" tab
    const repoTab = screen.getByTestId("add-skill-modal-tab-repo");
    await user.click(repoTab);

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

describe("AddSkillModal custom skill creation & editing", () => {
  it("renders custom skill form by default", () => {
    renderWithProviders(<AddSkillModal onClose={vi.fn()} />);

    expect(screen.getByTestId("custom-skill-name-input")).toBeInTheDocument();
    expect(
      screen.getByTestId("custom-skill-triggers-input"),
    ).toBeInTheDocument();
    const descInput = screen.getByTestId("custom-skill-description-input");
    expect(descInput).toBeInTheDocument();
    expect(descInput.tagName).toBe("TEXTAREA");
    expect(descInput).toHaveAttribute("rows", "4");
    expect(
      screen.getByTestId("custom-skill-content-input"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("save-custom-skill-button")).toBeInTheDocument();
  });

  it("validates required name and slug format", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddSkillModal onClose={vi.fn()} />);

    const saveButton = screen.getByTestId("save-custom-skill-button");

    // Attempt save with empty name
    await user.click(saveButton);
    expect(screen.getByTestId("custom-skill-name-error")).toBeInTheDocument();

    // Type invalid name (spaces / uppercase)
    const nameInput = screen.getByTestId("custom-skill-name-input");
    await user.type(nameInput, "Invalid Name!");
    await user.click(saveButton);
    expect(screen.getByTestId("custom-skill-name-error")).toBeInTheDocument();
  });

  it("validates required content", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddSkillModal onClose={vi.fn()} />);

    const nameInput = screen.getByTestId("custom-skill-name-input");
    await user.type(nameInput, "valid-name");

    const saveButton = screen.getByTestId("save-custom-skill-button");
    await user.click(saveButton);

    expect(
      screen.getByTestId("custom-skill-content-error"),
    ).toBeInTheDocument();
  });

  it("auto-populates name, description, and triggers from markdown frontmatter", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddSkillModal onClose={vi.fn()} />);

    const contentInput = screen.getByTestId("custom-skill-content-input");
    const markdownWithFrontmatter = `---
name: extracted-name
description: Extracted description
triggers:
  - extract-trig
---
# Body
Content here`;

    await user.type(contentInput, markdownWithFrontmatter);

    expect(screen.getByTestId("custom-skill-name-input")).toHaveValue(
      "extracted-name",
    );
    expect(screen.getByTestId("custom-skill-description-input")).toHaveValue(
      "Extracted description",
    );
    expect(screen.getByTestId("custom-skill-triggers-input")).toHaveValue(
      "extract-trig",
    );
  });

  it("saves a valid custom skill and closes the modal", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onSaved = vi.fn();

    renderWithProviders(<AddSkillModal onClose={onClose} onSaved={onSaved} />);

    await user.type(
      screen.getByTestId("custom-skill-name-input"),
      "my-custom-tool",
    );
    await user.type(
      screen.getByTestId("custom-skill-description-input"),
      "Helpful tool description",
    );
    await user.type(
      screen.getByTestId("custom-skill-triggers-input"),
      "tool, helper",
    );
    await user.type(
      screen.getByTestId("custom-skill-content-input"),
      "# My Custom Tool\nInstructions.",
    );

    await user.click(screen.getByTestId("save-custom-skill-button"));

    expect(onSaved).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "my-custom-tool",
        description: "Helpful tool description",
        source: "custom",
      }),
    );
    expect(onClose).toHaveBeenCalled();

    const stored = SkillsService.getCustomSkills();
    expect(stored).toHaveLength(1);
    expect(stored[0]?.name).toBe("my-custom-tool");
    expect(stored[0]?.triggers).toEqual(["tool", "helper"]);
  });

  it("pre-populates fields in edit mode and updates without creating duplicate", async () => {
    const user = userEvent.setup();
    const existingSkill: SkillInfo = {
      name: "editable-skill",
      type: "knowledge",
      source: "custom",
      description: "Original description",
      triggers: ["edit1"],
      content: "# Original Content",
    };

    SkillsService.saveCustomSkill(existingSkill);
    expect(SkillsService.getCustomSkills()).toHaveLength(1);

    const onClose = vi.fn();
    const onSaved = vi.fn();

    renderWithProviders(
      <AddSkillModal
        initialSkill={existingSkill}
        onClose={onClose}
        onSaved={onSaved}
      />,
    );

    const nameInput = screen.getByTestId("custom-skill-name-input");
    expect(nameInput).toHaveValue("editable-skill");
    expect(nameInput).toBeDisabled();

    const descInput = screen.getByTestId("custom-skill-description-input");
    expect(descInput).toHaveValue("Original description");

    await user.clear(descInput);
    await user.type(descInput, "Updated description");

    await user.click(screen.getByTestId("save-custom-skill-button"));

    expect(onSaved).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();

    const stored = SkillsService.getCustomSkills();
    expect(stored).toHaveLength(1);
    expect(stored[0]?.description).toBe("Updated description");
  });
});
