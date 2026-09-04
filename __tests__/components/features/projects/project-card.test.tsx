import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import type { ProjectSummary } from "#/api/projects-service/projects-types";
import { ProjectCard } from "#/components/features/projects/project-card";
import { I18nKey } from "#/i18n/declaration";

function makeProject(
  overrides: Partial<ProjectSummary> = {},
): ProjectSummary {
  return {
    id: "proj-1",
    name: "Agent Canvas",
    description: null,
    repo_url: "https://github.com/example/repo",
    local_path: "/tmp/agent-canvas",
    default_branch: "main",
    default_agent_profile: null,
    kanban_board_id: null,
    cost_cap: 40,
    status: "active",
    worktree_count: 2,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    ...overrides,
  };
}

describe("ProjectCard", () => {
  it("shows name, status, branch count, and cost", () => {
    renderWithProviders(<ProjectCard project={makeProject()} />);

    expect(screen.getByTestId("project-card-proj-1")).toHaveTextContent(
      "Agent Canvas",
    );
    expect(screen.getByTestId("project-card-status-proj-1")).toHaveTextContent(
      I18nKey.PROJECTS$STATUS_ACTIVE,
    );
    expect(screen.getByTestId("project-card-branches-proj-1")).toHaveTextContent(
      "2",
    );
    expect(screen.getByTestId("project-card-cost-proj-1")).toHaveTextContent(
      "$40.00",
    );
  });

  it("notifies the parent when selected", () => {
    const onSelect = vi.fn();
    renderWithProviders(
      <ProjectCard project={makeProject()} onSelect={onSelect} />,
    );

    fireEvent.click(screen.getByTestId("project-card-proj-1"));
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ id: "proj-1" }),
    );
  });
});
