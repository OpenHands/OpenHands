import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SecretsService } from "#/api/secrets-service";
import { SecretForm } from "#/components/features/settings/secrets-settings/secret-form";
import { renderWithProviders } from "../../../../../test-utils";

const renderEditForm = async () => {
  vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([
    { name: "API_KEY", description: "Demo secret" },
  ]);
  const updateSecret = vi
    .spyOn(SecretsService, "updateSecret")
    .mockResolvedValue(undefined);

  renderWithProviders(
    <SecretForm mode="edit" selectedSecret="API_KEY" onCancel={vi.fn()} />,
  );

  // The description default is applied once the secrets query settles.
  await waitFor(() =>
    expect(screen.getByTestId("description-input")).toHaveValue("Demo secret"),
  );

  return { updateSecret };
};

describe("SecretForm in edit mode", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("offers a blank value field labelled for preserving the existing value", async () => {
    // Arrange & Act
    await renderEditForm();

    // Assert
    expect(screen.getByTestId("value-input")).toHaveValue("");
    expect(
      screen.getByText("SECRETS$SECRET_VALUE_LEAVE_BLANK"),
    ).toBeInTheDocument();
  });

  it("keeps the stored value when the value field is left blank", async () => {
    // Arrange
    const { updateSecret } = await renderEditForm();

    // Act
    await userEvent.click(screen.getByTestId("submit-button"));

    // Assert
    await waitFor(() =>
      expect(updateSecret).toHaveBeenCalledWith(
        "API_KEY",
        "API_KEY",
        "Demo secret",
        undefined,
      ),
    );
  });

  it("overwrites the stored value with the one entered", async () => {
    // Arrange
    const { updateSecret } = await renderEditForm();

    // Act
    await userEvent.type(screen.getByTestId("value-input"), "sk-new-value");
    await userEvent.click(screen.getByTestId("submit-button"));

    // Assert
    await waitFor(() =>
      expect(updateSecret).toHaveBeenCalledWith(
        "API_KEY",
        "API_KEY",
        "Demo secret",
        "sk-new-value",
      ),
    );
  });
});
