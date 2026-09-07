import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import ProjectsService from "#/api/projects-service/projects-service.api";
import { projectDetailPath } from "#/api/projects-service/projects-constants";
import { I18nKey } from "#/i18n/declaration";
import { resetProjectsMockData } from "#/mocks/handlers";
import ProjectsPage from "#/routes/projects";

describe("ProjectsPage", () => {
  beforeEach(() => {
    resetProjectsMockData();
  });

  it("shows an empty state until a project is created", async () => {
    renderWithProviders(<ProjectsPage />);

    expect(await screen.findByTestId("projects-empty")).toHaveTextContent(
      I18nKey.PROJECTS$EMPTY,
    );
  });

  it("creates a project from the form and opens its detail view", async () => {
    const user = userEvent.setup();
    const navigate = vi.fn();
    renderWithProviders(<ProjectsPage />, { navigation: { navigate } });

    await user.click(screen.getByTestId("projects-create"));
    await user.type(screen.getByTestId("create-project-name"), "Alpha");
    await user.type(screen.getByTestId("create-project-repo-url"), "/tmp/src");
    await user.type(screen.getByTestId("create-project-cost-cap"), "12");
    await user.click(screen.getByTestId("create-project-submit"));

    await waitFor(() => {
      expect(navigate).toHaveBeenCalled();
    });
    const created = await ProjectsService.listProjects();
    expect(created[0].name).toBe("Alpha");
    expect(navigate).toHaveBeenCalledWith(projectDetailPath(created[0].id));
  });

  it("renders project cards on the list", async () => {
    await ProjectsService.createProject({ name: "Listed", cost_cap: 5 });
    renderWithProviders(<ProjectsPage />);

    expect(await screen.findByText("Listed")).toBeInTheDocument();
  });

  it("renders project detail when the path contains a project id", async () => {
    const created = await ProjectsService.createProject({ name: "Detail me" });
    renderWithProviders(<ProjectsPage />, {
      navigation: { currentPath: projectDetailPath(created.id) },
    });

    expect(await screen.findByTestId("project-detail")).toHaveTextContent(
      "Detail me",
    );
    expect(screen.getByTestId("worktree-panel")).toBeInTheDocument();
  });
});
