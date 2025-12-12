import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "i18next";
import { renderWithProviders } from "test-utils";
import { ConfirmDeleteModal } from "#/components/features/conversation-panel/confirm-delete-modal";
import { I18nKey } from "#/i18n/declaration";

const enTranslations = {
  [I18nKey.CONVERSATION$CONFIRM_DELETE]: "Confirm Delete",
  [I18nKey.CONVERSATION$DELETE_WARNING]:
    "Are you sure you want to delete this conversation? This action cannot be undone.",
  [I18nKey.CONVERSATION$DELETE_WARNING_WITH_TITLE]:
    'Are you sure you want to delete the "{{title}}" conversation? This action cannot be undone.',
  [I18nKey.ACTION$CONFIRM_DELETE]: "Confirm Delete",
  [I18nKey.BUTTON$CANCEL]: "Cancel",
};

describe("ConfirmDeleteModal", () => {
  const onConfirm = vi.fn();
  const onCancel = vi.fn();

  beforeEach(() => {
    onConfirm.mockClear();
    onCancel.mockClear();
    // Ensure translations are available for the tests
    i18n.addResourceBundle("en", "translation", enTranslations, true, true);
  });

  it("renders default warning when no title is provided", () => {
    renderWithProviders(
      <ConfirmDeleteModal onConfirm={onConfirm} onCancel={onCancel} />,
    );

    expect(
      screen.getByText(enTranslations[I18nKey.CONVERSATION$CONFIRM_DELETE]),
    ).toBeInTheDocument();
    expect(
      screen.getByText(enTranslations[I18nKey.CONVERSATION$DELETE_WARNING]),
    ).toBeInTheDocument();
  });

  it("renders interpolated warning when a title is provided", () => {
    const title = "Hello - Initial Greeting";
    renderWithProviders(
      <ConfirmDeleteModal
        onConfirm={onConfirm}
        onCancel={onCancel}
        conversationTitle={title}
      />,
    );

    expect(
      screen.getByText(
        enTranslations[I18nKey.CONVERSATION$DELETE_WARNING_WITH_TITLE].replace(
          "{{title}}",
          title,
        ),
      ),
    ).toBeInTheDocument();
  });

  it("fires confirm and cancel actions", async () => {
    renderWithProviders(
      <ConfirmDeleteModal onConfirm={onConfirm} onCancel={onCancel} />,
    );

    await userEvent.click(
      screen.getByText(enTranslations[I18nKey.ACTION$CONFIRM_DELETE]),
    );
    expect(onConfirm).toHaveBeenCalledTimes(1);

    await userEvent.click(
      screen.getByText(enTranslations[I18nKey.BUTTON$CANCEL]),
    );
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});

