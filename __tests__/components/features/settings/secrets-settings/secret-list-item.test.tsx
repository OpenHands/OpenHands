import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SecretListItem } from "#/components/features/settings/secrets-settings/secret-list-item";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        SECRETS$EDIT_SECRET_ARIA: params?.name ? `Edit ${params.name}` : "Edit",
        SECRETS$DELETE_SECRET_ARIA: params?.name
          ? `Delete ${params.name}`
          : "Delete",
      };
      return translations[key] || key;
    },
  }),
}));

describe("SecretListItem", () => {
  it("labels the edit and delete buttons through i18n, with the secret name interpolated", () => {
    // Regression: these were hardcoded English template literals
    // (`Edit ${title}` / `Delete ${title}`), the only place in this feature
    // area that bypassed t()/I18nKey.
    render(
      <table>
        <tbody>
          <SecretListItem
            title="OPENAI_API_KEY"
            description="Demo"
            onEdit={vi.fn()}
            onDelete={vi.fn()}
          />
        </tbody>
      </table>,
    );

    expect(screen.getByTestId("edit-secret-button")).toHaveAttribute(
      "aria-label",
      "Edit OPENAI_API_KEY",
    );
    expect(screen.getByTestId("delete-secret-button")).toHaveAttribute(
      "aria-label",
      "Delete OPENAI_API_KEY",
    );
  });
});
