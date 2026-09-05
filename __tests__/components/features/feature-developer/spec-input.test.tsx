import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { SpecInput } from "#/components/features/feature-developer/spec-input";
import { I18nKey } from "#/i18n/declaration";
import type { ProjectSummary } from "#/api/projects-service/projects-types";

const PROJECT: ProjectSummary = {
  id: "proj-1",
  name: "Alpha",
  description: null,
  repo_url: null,
  local_path: "/tmp/alpha",
  default_branch: "main",
  default_agent_profile: null,
  kanban_board_id: null,
  cost_cap: 10,
  status: "idle",
  worktree_count: 0,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

describe("SpecInput", () => {
  it("shows validation errors when started without a project or spec", async () => {
    const user = userEvent.setup();
    const onStart = vi.fn();
    renderWithProviders(<SpecInput projects={[PROJECT]} onStart={onStart} />);

    await user.click(screen.getByTestId("feature-dev-start"));

    expect(screen.getByTestId("feature-dev-spec-error")).toHaveTextContent(
      I18nKey.FEATURE_DEV$VALIDATION_SPEC,
    );
    expect(screen.getByTestId("feature-dev-project-error")).toHaveTextContent(
      I18nKey.FEATURE_DEV$VALIDATION_PROJECT,
    );
    expect(onStart).not.toHaveBeenCalled();
  });

  it("starts a run when project and spec are provided", async () => {
    const user = userEvent.setup();
    const onStart = vi.fn();
    renderWithProviders(<SpecInput projects={[PROJECT]} onStart={onStart} />);

    await user.click(screen.getByLabelText(I18nKey.FEATURE_DEV$PROJECT));
    await user.click(await screen.findByText("Alpha"));
    await user.type(screen.getByTestId("feature-dev-spec"), "Build login");
    await user.click(screen.getByTestId("feature-dev-continue-on-failure"));
    await user.click(screen.getByTestId("feature-dev-start"));

    expect(onStart).toHaveBeenCalledWith({
      project_id: "proj-1",
      spec_text: "Build login",
      max_concurrent_agents: 1,
      continue_on_failure: true,
    });
  });
});
