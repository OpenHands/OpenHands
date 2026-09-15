import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { SkillsModal } from "#/components/features/conversation-panel/skills-modal";

vi.mock("#/hooks/query/use-conversation-skills", () => ({
  useConversationSkills: () => ({
    data: [],
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
    isRefetching: false,
  }),
}));

vi.mock("#/hooks/use-skill-enablement", () => ({
  useSkillEnabledFilter: () => () => true,
}));

describe("SkillsModal", () => {
  it("gives the dialog an accessible name matching its visible title", () => {
    // Regression: ModalBackdrop always sets role="dialog" aria-modal="true"
    // but only gets an accessible name when the caller passes aria-label —
    // this modal never did, leaving screen readers an unnamed dialog.
    renderWithProviders(<SkillsModal onClose={vi.fn()} />);

    expect(
      screen.getByRole("dialog", { name: "SKILLS_MODAL$TITLE" }),
    ).toBeInTheDocument();
  });
});
