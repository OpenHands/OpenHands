import { beforeEach, describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { MacrosManager } from "#/components/features/settings/macros/macros-manager";
import { createDefaultMacros } from "#/utils/macros";
import { useMacrosStore } from "#/stores/macros-store";

function renderMacrosManager() {
  return renderWithProviders(<MacrosManager />);
}

describe("MacrosManager", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useMacrosStore.setState({ macros: createDefaultMacros() });
  });

  it("lists saved macros and leftover catalog entries", () => {
    renderMacrosManager();

    expect(screen.getByTestId("macros-settings-screen")).toBeInTheDocument();
    expect(
      screen.getByTestId("macro-item-INCREASE_TEST_COVERAGE"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("add-catalog-macro-ADD_DOCS"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("macros-automations-link")).toHaveAttribute(
      "href",
      "/automations",
    );
  });

  it("adds a custom macro from the form", async () => {
    const user = userEvent.setup();
    renderMacrosManager();

    await user.click(screen.getByTestId("add-macro-button"));
    await user.type(screen.getByTestId("macro-title-input"), "Triage bugs");
    await user.type(
      screen.getByTestId("macro-prompt-input"),
      "List open bugs first.",
    );
    await user.click(screen.getByTestId("save-macro-button"));

    expect(screen.getByText("Triage bugs")).toBeInTheDocument();
    expect(
      useMacrosStore
        .getState()
        .macros.some((macro) => macro.title === "Triage bugs"),
    ).toBe(true);
  });

  it("edits and deletes a saved macro", async () => {
    const user = userEvent.setup();
    renderMacrosManager();

    await user.click(screen.getByTestId("edit-macro-INCREASE_TEST_COVERAGE"));
    const titleInput = screen.getByTestId("macro-title-input");
    await user.clear(titleInput);
    await user.type(titleInput, "Coverage sprint");
    await user.click(screen.getByTestId("save-macro-button"));

    expect(screen.getByText("Coverage sprint")).toBeInTheDocument();

    await user.click(screen.getByTestId("delete-macro-INCREASE_TEST_COVERAGE"));
    await user.click(screen.getByTestId("confirm-button"));

    expect(
      screen.queryByTestId("macro-item-INCREASE_TEST_COVERAGE"),
    ).not.toBeInTheDocument();
  });

  it("adds a catalog macro into the saved list", async () => {
    const user = userEvent.setup();
    renderMacrosManager();

    await user.click(screen.getByTestId("add-catalog-macro-ADD_DOCS"));

    expect(screen.getByTestId("macro-item-ADD_DOCS")).toBeInTheDocument();
    expect(
      screen.queryByTestId("add-catalog-macro-ADD_DOCS"),
    ).not.toBeInTheDocument();
  });
});
