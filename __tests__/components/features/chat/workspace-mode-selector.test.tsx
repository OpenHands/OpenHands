import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { WorkspaceModeSelector } from "#/components/features/chat/workspace-mode-selector";

describe("WorkspaceModeSelector", () => {
  it("opens the menu and calls onChange when an option is selected", () => {
    const onChange = vi.fn();
    render(
      <WorkspaceModeSelector
        value="local_repo"
        backendKind="local"
        onChange={onChange}
      />,
    );

    // Open the menu (trigger renders the active mode via WorkspaceModeIcon).
    fireEvent.click(screen.getByTestId("workspace-mode-selector"));
    fireEvent.click(
      screen.getByTestId("workspace-mode-selector-option-new_worktree"),
    );

    expect(onChange).toHaveBeenCalledWith("new_worktree");
  });
});
