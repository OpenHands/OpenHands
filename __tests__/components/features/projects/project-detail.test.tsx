import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { KANBAN_PATH } from "#/api/kanban-service/kanban-constants";
import type { Project } from "#/api/projects-service/projects-types";
import { ProjectDetail } from "#/components/features/projects/project-detail";
import { I18nKey } from "#/i18n/declaration";

function makeProject(overrides: Partial<Project> = {}): Project {
  return {
    id: "proj-1",
    name: "Agent Canvas",
    description: "Spaces",
    repo_url: null,
    local_path: "/tmp/agent-canvas",
    default_branch: "main",
    default_agent_profile: null,
    kanban_board_id: "board-1",
    cost_cap: 25,
    status: "idle",
    worktree_count: 1,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    worktrees: [
      {
        id: "wt-1",
        project_id: "proj-1",
        branch_name: "main",
        path: "/tmp/agent-canvas/.worktrees/main",
        status: "idle",
        agent_session_id: null,
        created_at: "2026-09-01T00:00:00Z",
        updated_at: "2026-09-01T00:00:00Z",
      },
    ],
    ...overrides,
  };
}

describe("ProjectDetail", () => {
  it("renders metadata, kanban link, and worktrees", () => {
    const navigate = vi.fn();
    renderWithProviders(<ProjectDetail project={makeProject()} />, {
      navigation: { navigate },
    });

    expect(screen.getByTestId("project-detail")).toHaveTextContent(
      "Agent Canvas",
    );
    expect(screen.getByTestId("project-metadata")).toHaveTextContent("/tmp/agent-canvas");
    expect(screen.getByTestId("worktree-row-wt-1")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("project-open-kanban"));
    expect(navigate).toHaveBeenCalledWith(KANBAN_PATH);
  });

  it("shows an empty kanban state when no board is linked", () => {
    renderWithProviders(
      <ProjectDetail project={makeProject({ kanban_board_id: null })} />,
    );

    expect(screen.getByTestId("project-no-kanban")).toHaveTextContent(
      I18nKey.PROJECTS$NO_KANBAN,
    );
  });
});
