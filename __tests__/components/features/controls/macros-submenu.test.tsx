import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { MacrosSubmenu } from "#/components/features/controls/macros-submenu";
import { REPO_SUGGESTIONS } from "#/utils/suggestions/repo-suggestions";
import { createDefaultMacros } from "#/utils/macros";
import { useMacrosStore } from "#/stores/macros-store";
import { useConversationStore } from "#/stores/conversation-store";

describe("MacrosSubmenu", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useMacrosStore.setState({ macros: createDefaultMacros() });
    useConversationStore.setState({ messageToSend: null });
  });

  it("inserts a saved macro into the chat input", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderWithProviders(<MacrosSubmenu onClose={onClose} />);

    await user.click(screen.getByTestId("macro-INCREASE_TEST_COVERAGE"));

    expect(useConversationStore.getState().messageToSend?.text).toBe(
      REPO_SUGGESTIONS.INCREASE_TEST_COVERAGE,
    );
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("links to the macros settings page", () => {
    renderWithProviders(<MacrosSubmenu onClose={vi.fn()} />);

    expect(screen.getByTestId("manage-macros-button")).toHaveAttribute(
      "href",
      "/settings/macros",
    );
  });

  it("renders a newly added custom macro", () => {
    useMacrosStore.getState().addMacro({
      title: "Triage bugs",
      prompt: "List open bugs first.",
    });
    const custom = useMacrosStore
      .getState()
      .macros.find((macro) => macro.title === "Triage bugs")!;

    renderWithProviders(<MacrosSubmenu onClose={vi.fn()} />);

    expect(screen.getByTestId(`macro-${custom.id}`)).toHaveTextContent(
      "Triage bugs",
    );
  });
});
