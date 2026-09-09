import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProjectInitForm } from "#/components/features/kanban/project-init-form";
import { renderWithProviders } from "test-utils";
import { I18nKey } from "#/i18n/declaration";
import { KANBAN_PATH } from "#/api/kanban-service/kanban-constants";

describe("ProjectInitForm", () => {
  it("previews suggested cards then creates a board", async () => {
    const navigate = vi.fn();
    renderWithProviders(<ProjectInitForm />, {
      navigation: { navigate, currentPath: "/project-init" },
    });

    fireEvent.change(screen.getByTestId("project-init-spec"), {
      target: { value: "Build login" },
    });
    fireEvent.click(screen.getByTestId("project-init-scan"));

    await waitFor(() => {
      expect(
        screen.getByTestId("project-init-suggested-card"),
      ).toHaveTextContent("Build login");
    });

    fireEvent.click(screen.getByTestId("project-init-create"));

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith(KANBAN_PATH);
    });
  });

  it("shows an empty preview before scanning", () => {
    renderWithProviders(<ProjectInitForm />);
    expect(screen.getByTestId("project-init-empty")).toHaveTextContent(
      I18nKey.PROJECT_INIT$NO_CARDS,
    );
  });
});
