import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient } from "@tanstack/react-query";
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

  it("reveals and hides only the entered replacement without submitting", async () => {
    const user = userEvent.setup();
    const { updateSecret } = await renderEditForm();
    const input = screen.getByTestId("value-input");

    await user.type(input, "dummy-replacement");
    expect(input).toHaveAttribute("type", "password");
    await user.click(
      screen.getByRole("button", { name: "SECRETS$SHOW_VALUE" }),
    );
    expect(input).toHaveAttribute("type", "text");
    expect(input).toHaveValue("dummy-replacement");
    expect(updateSecret).not.toHaveBeenCalled();

    await user.click(
      screen.getByRole("button", { name: "SECRETS$HIDE_VALUE" }),
    );
    expect(input).toHaveAttribute("type", "password");
    await user.click(screen.getByTestId("submit-button"));
    await waitFor(() =>
      expect(updateSecret).toHaveBeenCalledWith(
        "API_KEY",
        "API_KEY",
        "Demo secret",
        "dummy-replacement",
      ),
    );
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

    // Act — submit is dirty-gated, so change description while leaving value blank
    await userEvent.type(
      screen.getByTestId("description-input"),
      " (unchanged value)",
    );
    await userEvent.click(screen.getByTestId("submit-button"));

    // Assert
    await waitFor(() =>
      expect(updateSecret).toHaveBeenCalledWith(
        "API_KEY",
        "API_KEY",
        "Demo secret (unchanged value)",
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

describe("SecretForm save lifecycle", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // The chat unmounts the secret-request form when its virtual row scrolls out
  // of the overscan while a save is in flight. TanStack Query drops per-call
  // `mutate` callbacks with the unmounted observer, so the cache invalidation
  // must ride the mutation itself; otherwise Settings keeps the stale secrets
  // list for the five-minute `staleTime`.
  it("invalidates the secrets cache even when the form unmounts before the save resolves", async () => {
    const invalidate = vi.spyOn(QueryClient.prototype, "invalidateQueries");
    let resolveSave: () => void = () => {};
    vi.spyOn(SecretsService, "createSecret").mockReturnValue(
      new Promise<void>((resolve) => {
        resolveSave = resolve;
      }),
    );

    const user = userEvent.setup();
    const { unmount } = renderWithProviders(
      <SecretForm mode="add" selectedSecret={null} onCancel={vi.fn()} />,
    );

    await user.type(screen.getByTestId("name-input"), "NEW_SECRET");
    await user.type(screen.getByTestId("value-input"), "value");
    await user.click(screen.getByTestId("submit-button"));

    expect(invalidate).not.toHaveBeenCalled();
    unmount();

    await act(async () => {
      resolveSave();
      await Promise.resolve();
    });

    await waitFor(() =>
      expect(invalidate).toHaveBeenCalledWith({
        queryKey: ["secrets-search"],
      }),
    );
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["secrets"] });
  });
});
