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

  it("offers docker only when passed as an option", () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <WorkspaceModeSelector
        value="local_repo"
        backendKind="local"
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByTestId("workspace-mode-selector"));
    expect(
      screen.queryByTestId("workspace-mode-selector-option-docker_container"),
    ).toBeNull();

    rerender(
      <WorkspaceModeSelector
        value="local_repo"
        backendKind="local"
        options={["local_repo", "new_worktree", "docker_container"]}
        onChange={onChange}
      />,
    );
    fireEvent.click(
      screen.getByTestId("workspace-mode-selector-option-docker_container"),
    );
    expect(onChange).toHaveBeenCalledWith("docker_container");
  });
});
