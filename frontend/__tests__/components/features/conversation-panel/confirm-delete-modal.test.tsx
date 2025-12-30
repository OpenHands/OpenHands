import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { ConfirmDeleteModal } from "#/components/features/conversation-panel/confirm-delete-modal";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({
    t: (key: string, options?: { title?: string }) =>
      options?.title ? `Delete "${options.title}"?` : key,
  }),
}));

describe("ConfirmDeleteModal", () => {
  it("should display the conversation title", () => {
    renderWithProviders(
      <ConfirmDeleteModal
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
        conversationTitle="My Test Conversation"
      />,
    );

    expect(screen.getByText(/My Test Conversation/)).toBeInTheDocument();
  });
});
