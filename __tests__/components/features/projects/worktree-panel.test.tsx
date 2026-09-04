import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import type { ProjectWorktree } from "#/api/projects-service/projects-types";
import { WorktreePanel } from "#/components/features/projects/worktree-panel";
import { I18nKey } from "#/i18n/declaration";

function makeWorktree(
  overrides: Partial<ProjectWorktree> = {},
): ProjectWorktree {
  return {
    id: "wt-1",
    project_id: "proj-1",
    branch_name: "feature/spaces",
    path: "/tmp/repo/.worktrees/feature--spaces",
    status: "working",
    agent_session_id: "sess-1",
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    ...overrides,
  };
}

describe("WorktreePanel", () => {
  it("lists worktrees with branch, status, and assigned agent", () => {
    renderWithProviders(<WorktreePanel worktrees={[makeWorktree()]} />);

    expect(screen.getByTestId("worktree-row-wt-1")).toHaveTextContent(
      "feature/spaces",
    );
    expect(screen.getByTestId("worktree-status-wt-1")).toHaveTextContent(
      I18nKey.PROJECTS$WORKTREE_WORKING,
    );
    expect(screen.getByTestId("worktree-row-wt-1")).toHaveTextContent("sess-1");
  });

  it("creates a worktree from the branch form", () => {
    const onAdd = vi.fn();
    renderWithProviders(<WorktreePanel worktrees={[]} onAdd={onAdd} />);

    expect(screen.getByTestId("worktree-empty")).toHaveTextContent(
      I18nKey.PROJECTS$NO_WORKTREES,
    );
    fireEvent.change(screen.getByTestId("worktree-branch-name"), {
      target: { value: "agent/fix" },
    });
    fireEvent.click(screen.getByTestId("worktree-add"));
    expect(onAdd).toHaveBeenCalledWith("agent/fix");
  });

  it("assigns and removes a worktree", () => {
    const onAssign = vi.fn();
    const onRemove = vi.fn();
    renderWithProviders(
      <WorktreePanel
        worktrees={[makeWorktree({ agent_session_id: null, status: "idle" })]}
        onAssign={onAssign}
        onRemove={onRemove}
      />,
    );

    fireEvent.change(screen.getByTestId("worktree-session-id"), {
      target: { value: "sess-9" },
    });
    fireEvent.click(screen.getByTestId("worktree-assign-wt-1"));
    expect(onAssign).toHaveBeenCalledWith("wt-1", "sess-9");
    fireEvent.click(screen.getByTestId("worktree-remove-wt-1"));
    expect(onRemove).toHaveBeenCalledWith("wt-1");
  });
});
