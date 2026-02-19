import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { createRoutesStub, Outlet } from "react-router";
import SecretsSettingsScreen from "#/routes/secrets-settings";
import { SecretsService } from "#/api/secrets-service";
import { GetSecretsResponse } from "#/api/secrets-service.types";
import SettingsService from "#/api/settings-service/settings-service.api";
import OptionService from "#/api/option-service/option-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";

const MOCK_GET_SECRETS_RESPONSE: GetSecretsResponse["custom_secrets"] = [
  {
    name: "My_Secret_1",
    description: "My first secret",
  },
  {
    name: "My_Secret_2",
    description: "My second secret",
  },
];

const renderSecretsSettings = () => {
  const RouterStub = createRoutesStub([
    {
      Component: () => <Outlet />,
      path: "/settings",
      children: [
        {
          Component: SecretsSettingsScreen,
          path: "/settings/secrets",
        },
        {
          Component: () => <div data-testid="git-settings-screen" />,
          path: "/settings/integrations",
        },
      ],
    },
  ]);

  return render(<RouterStub initialEntries={["/settings/secrets"]} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        {children}
      </QueryClientProvider>
    ),
  });
};

beforeEach(() => {
  const getConfigSpy = vi.spyOn(OptionService, "getConfig");
  // @ts-expect-error - only return the config we need
  getConfigSpy.mockResolvedValue({
    app_mode: "oss",
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Content", () => {
  it("should render the secrets settings screen", () => {
    renderSecretsSettings();
    screen.getByTestId("secrets-settings-screen");
  });

  it("should NOT render a button to connect with git if they havent already in oss", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    // @ts-expect-error - only return the config we need
    getConfigSpy.mockResolvedValue({
      app_mode: "oss",
    });
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      provider_tokens_set: {},
    });

    renderSecretsSettings();

    expect(getConfigSpy).toHaveBeenCalled();
    await waitFor(() => expect(getSecretsSpy).toHaveBeenCalled());
    expect(screen.queryByTestId("connect-git-button")).not.toBeInTheDocument();
  });

  it("should render add secret button in saas mode", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    // @ts-expect-error - only return the config we need
    getConfigSpy.mockResolvedValue({
      app_mode: "saas",
    });

    renderSecretsSettings();

    // In SAAS mode, getSecrets is called and add secret button should be available
    await waitFor(() => expect(getSecretsSpy).toHaveBeenCalled());
    const button = await screen.findByTestId("add-secret-button");
    expect(button).toBeInTheDocument();
    expect(screen.queryByTestId("connect-git-button")).not.toBeInTheDocument();
  });

  it("should render an empty table when there are no existing secrets", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    getSecretsSpy.mockResolvedValue([]);
    renderSecretsSettings();

    // Should show the add secret button
    await screen.findByTestId("add-secret-button");

    // Should show an empty table with headers but no secret items
    expect(screen.queryAllByTestId("secret-item")).toHaveLength(0);

    // Should still show the table headers
    expect(screen.getByText("SETTINGS$NAME")).toBeInTheDocument();
    expect(screen.getByText("SECRETS$DESCRIPTION")).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$ACTIONS")).toBeInTheDocument();
  });

  it("should render existing secrets", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    getSecretsSpy.mockResolvedValue(MOCK_GET_SECRETS_RESPONSE);
    renderSecretsSettings();

    const secrets = await screen.findAllByTestId("secret-item");
    expect(secrets).toHaveLength(2);
  });
});

describe("Secret actions", () => {
  it("should create a new secret", async () => {
    const createSecretSpy = vi.spyOn(SecretsService, "createSecret");
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    createSecretSpy.mockResolvedValue(true);
    renderSecretsSettings();

    // render form & hide items
    expect(screen.queryByTestId("add-secret-form")).not.toBeInTheDocument();
    const button = await screen.findByTestId("add-secret-button");
    await userEvent.click(button);

    const secretForm = screen.getByTestId("add-secret-form");
    const secrets = screen.queryAllByTestId("secret-item");

    expect(screen.queryByTestId("add-secret-button")).not.toBeInTheDocument();
    expect(secretForm).toBeInTheDocument();
    expect(secrets).toHaveLength(0);

    // enter details
    const nameInput = within(secretForm).getByTestId("name-input");
    const valueInput = within(secretForm).getByTestId("value-input");
    const descriptionInput =
      within(secretForm).getByTestId("description-input");

    const submitButton = within(secretForm).getByTestId("submit-button");

    vi.clearAllMocks(); // reset mocks to check for upcoming calls

    await userEvent.type(nameInput, "My_Custom_Secret");
    await userEvent.type(valueInput, "my-custom-secret-value");
    await userEvent.type(descriptionInput, "My custom secret description");

    await userEvent.click(submitButton);

    // make POST request
    expect(createSecretSpy).toHaveBeenCalledWith(
      "My_Custom_Secret",
      "my-custom-secret-value",
      "My custom secret description",
    );

    // hide form & render items
    expect(screen.queryByTestId("add-secret-form")).not.toBeInTheDocument();
    expect(getSecretsSpy).toHaveBeenCalled();
  });

  it("should edit a secret", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    const updateSecretSpy = vi.spyOn(SecretsService, "updateSecret");
    getSecretsSpy.mockResolvedValue(MOCK_GET_SECRETS_RESPONSE);
    updateSecretSpy.mockResolvedValue(true);
    renderSecretsSettings();

    // render edit button within a secret list item
    const secrets = await screen.findAllByTestId("secret-item");
    const firstSecret = within(secrets[0]);
    const editButton = firstSecret.getByTestId("edit-secret-button");

    await userEvent.click(editButton);

    // render edit form
    const editForm = screen.getByTestId("edit-secret-form");

    expect(screen.queryByTestId("add-secret-button")).not.toBeInTheDocument();
    expect(editForm).toBeInTheDocument();
    expect(screen.queryAllByTestId("secret-item")).toHaveLength(0);

    // enter details
    const nameInput = within(editForm).getByTestId("name-input");
    const descriptionInput = within(editForm).getByTestId("description-input");
    const submitButton = within(editForm).getByTestId("submit-button");

    // should not show value input
    const valueInput = within(editForm).queryByTestId("value-input");
    expect(valueInput).not.toBeInTheDocument();

    expect(nameInput).toHaveValue("My_Secret_1");
    expect(descriptionInput).toHaveValue("My first secret");

    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, "My_Edited_Secret");

    await userEvent.clear(descriptionInput);
    await userEvent.type(descriptionInput, "My edited secret description");

    await userEvent.click(submitButton);

    // make POST request
    expect(updateSecretSpy).toHaveBeenCalledWith(
      "My_Secret_1",
      "My_Edited_Secret",
      "My edited secret description",
    );

    // hide form
    expect(screen.queryByTestId("edit-secret-form")).not.toBeInTheDocument();

    // optimistic update
    const updatedSecrets = await screen.findAllByTestId("secret-item");
    expect(updatedSecrets).toHaveLength(2);
    expect(updatedSecrets[0]).toHaveTextContent(/my_edited_secret/i);
  });

  it("should be able to cancel the create or edit form", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    getSecretsSpy.mockResolvedValue(MOCK_GET_SECRETS_RESPONSE);
    renderSecretsSettings();

    // render form & hide items
    expect(screen.queryByTestId("add-secret-form")).not.toBeInTheDocument();
    const button = await screen.findByTestId("add-secret-button");
    await userEvent.click(button);
    const secretForm = screen.getByTestId("add-secret-form");
    expect(secretForm).toBeInTheDocument();

    // cancel button
    const cancelButton = within(secretForm).getByTestId("cancel-button");
    await userEvent.click(cancelButton);
    expect(screen.queryByTestId("add-secret-form")).not.toBeInTheDocument();
    expect(screen.queryByTestId("add-secret-button")).toBeInTheDocument();

    // render edit button within a secret list item
    const secrets = await screen.findAllByTestId("secret-item");
    const firstSecret = within(secrets[0]);
    const editButton = firstSecret.getByTestId("edit-secret-button");
    await userEvent.click(editButton);

    // render edit form
    const editForm = screen.getByTestId("edit-secret-form");
    expect(editForm).toBeInTheDocument();
    expect(screen.queryAllByTestId("secret-item")).toHaveLength(0);

    // cancel button
    const cancelEditButton = within(editForm).getByTestId("cancel-button");
    await userEvent.click(cancelEditButton);
    expect(screen.queryByTestId("edit-secret-form")).not.toBeInTheDocument();
    expect(screen.queryAllByTestId("secret-item")).toHaveLength(2);
  });

  it("should undo the optimistic update if the request fails", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    const updateSecretSpy = vi.spyOn(SecretsService, "updateSecret");
    getSecretsSpy.mockResolvedValue(MOCK_GET_SECRETS_RESPONSE);
    updateSecretSpy.mockRejectedValue(new Error("Failed to update secret"));
    renderSecretsSettings();

    // render edit button within a secret list item
    const secrets = await screen.findAllByTestId("secret-item");
    const firstSecret = within(secrets[0]);
    const editButton = firstSecret.getByTestId("edit-secret-button");

    await userEvent.click(editButton);

    // render edit form
    const editForm = screen.getByTestId("edit-secret-form");

    expect(editForm).toBeInTheDocument();
    expect(screen.queryAllByTestId("secret-item")).toHaveLength(0);

    // enter details
    const nameInput = within(editForm).getByTestId("name-input");
    const submitButton = within(editForm).getByTestId("submit-button");

    // should not show value input
    const valueInput = within(editForm).queryByTestId("value-input");
    expect(valueInput).not.toBeInTheDocument();

    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, "My_Edited_Secret");
    await userEvent.click(submitButton);

    // make POST request
    expect(updateSecretSpy).toHaveBeenCalledWith(
      "My_Secret_1",
      "My_Edited_Secret",
      "My first secret",
    );

    // hide form
    expect(screen.queryByTestId("edit-secret-form")).not.toBeInTheDocument();

    // no optimistic update
    const updatedSecrets = await screen.findAllByTestId("secret-item");
    expect(updatedSecrets).toHaveLength(2);
    expect(updatedSecrets[0]).not.toHaveTextContent(/my edited secret/i);
  });

  it("should remove the secret from the list after deletion", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    const deleteSecretSpy = vi.spyOn(SecretsService, "deleteSecret");
    getSecretsSpy.mockResolvedValue(MOCK_GET_SECRETS_RESPONSE);
    deleteSecretSpy.mockResolvedValue(true);
    renderSecretsSettings();

    // render delete button within a secret list item
    const secrets = await screen.findAllByTestId("secret-item");
    const secondSecret = within(secrets[1]);
    const deleteButton = secondSecret.getByTestId("delete-secret-button");
    await userEvent.click(deleteButton);

    // confirmation modal
    const confirmationModal = screen.getByTestId("confirmation-modal");
    const confirmButton =
      within(confirmationModal).getByTestId("confirm-button");
    await userEvent.click(confirmButton);

    // make DELETE request
    expect(deleteSecretSpy).toHaveBeenCalledWith("My_Secret_2");
    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();

    // optimistic update
    expect(screen.queryAllByTestId("secret-item")).toHaveLength(1);
    expect(screen.queryByText("My_Secret_2")).not.toBeInTheDocument();
  });

  it("should be able to cancel the delete confirmation modal", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    const deleteSecretSpy = vi.spyOn(SecretsService, "deleteSecret");
    getSecretsSpy.mockResolvedValue(MOCK_GET_SECRETS_RESPONSE);
    deleteSecretSpy.mockResolvedValue(true);
    renderSecretsSettings();

    // render delete button within a secret list item
    const secrets = await screen.findAllByTestId("secret-item");
    const secondSecret = within(secrets[1]);
    const deleteButton = secondSecret.getByTestId("delete-secret-button");
    await userEvent.click(deleteButton);

    // confirmation modal
    const confirmationModal = screen.getByTestId("confirmation-modal");
    const cancelButton = within(confirmationModal).getByTestId("cancel-button");
    await userEvent.click(cancelButton);

    // no DELETE request
    expect(deleteSecretSpy).not.toHaveBeenCalled();
    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();
    expect(screen.queryAllByTestId("secret-item")).toHaveLength(2);
  });

  it("should revert the optimistic update if the request fails", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    const deleteSecretSpy = vi.spyOn(SecretsService, "deleteSecret");
    getSecretsSpy.mockResolvedValue(MOCK_GET_SECRETS_RESPONSE);
    deleteSecretSpy.mockRejectedValue(new Error("Failed to delete secret"));
    renderSecretsSettings();

    // render delete button within a secret list item
    const secrets = await screen.findAllByTestId("secret-item");
    const secondSecret = within(secrets[1]);
    const deleteButton = secondSecret.getByTestId("delete-secret-button");
    await userEvent.click(deleteButton);

    // confirmation modal
    const confirmationModal = screen.getByTestId("confirmation-modal");
    const confirmButton =
      within(confirmationModal).getByTestId("confirm-button");
    await userEvent.click(confirmButton);

    // make DELETE request
    expect(deleteSecretSpy).toHaveBeenCalledWith("My_Secret_2");
    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();

    // optimistic update
    expect(screen.queryAllByTestId("secret-item")).toHaveLength(2);
    expect(screen.queryByText("My_Secret_2")).toBeInTheDocument();
  });

  it("should hide the table and add button when in form view", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    getSecretsSpy.mockResolvedValue([]);
    renderSecretsSettings();

    // Initially should show the add button and table
    const button = await screen.findByTestId("add-secret-button");
    expect(screen.getByText("SETTINGS$NAME")).toBeInTheDocument(); // table header

    await userEvent.click(button);

    // When in form view, should hide the add button and table
    const secretForm = screen.getByTestId("add-secret-form");
    expect(secretForm).toBeInTheDocument();
    expect(screen.queryByTestId("add-secret-button")).not.toBeInTheDocument();
    expect(screen.queryByText("SETTINGS$NAME")).not.toBeInTheDocument(); // table header should be hidden
  });

  it("should not allow spaces in secret names", async () => {
    const createSecretSpy = vi.spyOn(SecretsService, "createSecret");
    renderSecretsSettings();

    // render form & hide items
    expect(screen.queryByTestId("add-secret-form")).not.toBeInTheDocument();
    const button = await screen.findByTestId("add-secret-button");
    await userEvent.click(button);

    const secretForm = screen.getByTestId("add-secret-form");
    expect(secretForm).toBeInTheDocument();

    // enter details
    const nameInput = within(secretForm).getByTestId("name-input");
    const valueInput = within(secretForm).getByTestId("value-input");
    const submitButton = within(secretForm).getByTestId("submit-button");

    await userEvent.type(nameInput, "My Custom Secret With Spaces");
    await userEvent.type(valueInput, "my-custom-secret-value");
    await userEvent.click(submitButton);

    // make POST request
    expect(createSecretSpy).not.toHaveBeenCalled();

    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, "MyCustomSecret");
    await userEvent.click(submitButton);

    expect(createSecretSpy).toHaveBeenCalledWith(
      "MyCustomSecret",
      "my-custom-secret-value",
      undefined,
    );
  });

  it("should not allow existing secret names", async () => {
    const createSecretSpy = vi.spyOn(SecretsService, "createSecret");
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    getSecretsSpy.mockResolvedValue(MOCK_GET_SECRETS_RESPONSE.slice(0, 1));
    renderSecretsSettings();

    // render form & hide items
    expect(screen.queryByTestId("add-secret-form")).not.toBeInTheDocument();
    const button = await screen.findByTestId("add-secret-button");
    await userEvent.click(button);

    const secretForm = screen.getByTestId("add-secret-form");
    expect(secretForm).toBeInTheDocument();

    // enter details
    const nameInput = within(secretForm).getByTestId("name-input");
    const valueInput = within(secretForm).getByTestId("value-input");
    const submitButton = within(secretForm).getByTestId("submit-button");

    await userEvent.type(nameInput, "My_Secret_1");
    await userEvent.type(valueInput, "my-custom-secret-value");
    await userEvent.click(submitButton);

    // make POST request
    expect(createSecretSpy).not.toHaveBeenCalled();
    expect(
      screen.queryByText("SECRETS$SECRET_ALREADY_EXISTS"),
    ).toBeInTheDocument();

    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, "My_Custom_Secret");

    await userEvent.clear(valueInput);
    await userEvent.type(valueInput, "my-custom-secret-value");

    await userEvent.click(submitButton);

    expect(createSecretSpy).toHaveBeenCalledWith(
      "My_Custom_Secret",
      "my-custom-secret-value",
      undefined,
    );
    expect(
      screen.queryByText("SECRETS$SECRET_VALUE_REQUIRED"),
    ).not.toBeInTheDocument();
  });

  it("should not submit whitespace secret names or values", async () => {
    const createSecretSpy = vi.spyOn(SecretsService, "createSecret");
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    getSecretsSpy.mockResolvedValue([]);
    renderSecretsSettings();

    // render form & hide items
    expect(screen.queryByTestId("add-secret-form")).not.toBeInTheDocument();
    const button = await screen.findByTestId("add-secret-button");
    await userEvent.click(button);

    const secretForm = screen.getByTestId("add-secret-form");
    expect(secretForm).toBeInTheDocument();

    // enter details
    const nameInput = within(secretForm).getByTestId("name-input");
    const valueInput = within(secretForm).getByTestId("value-input");
    const submitButton = within(secretForm).getByTestId("submit-button");

    await userEvent.type(nameInput, "   ");
    await userEvent.type(valueInput, "my-custom-secret-value");
    await userEvent.click(submitButton);

    // make POST request
    expect(createSecretSpy).not.toHaveBeenCalled();

    await userEvent.clear(nameInput);
    await userEvent.type(nameInput, "My_Custom_Secret");

    await userEvent.clear(valueInput);
    await userEvent.type(valueInput, "   ");

    await userEvent.click(submitButton);

    expect(createSecretSpy).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(
        screen.queryByText("SECRETS$SECRET_VALUE_REQUIRED"),
      ).toBeInTheDocument();
    });
  });

  it("should not reset ipout values on an invalid submit", async () => {
    const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
    const createSecretSpy = vi.spyOn(SecretsService, "createSecret");
    getSecretsSpy.mockResolvedValue(MOCK_GET_SECRETS_RESPONSE);

    renderSecretsSettings();

    // render form & hide items
    expect(screen.queryByTestId("add-secret-form")).not.toBeInTheDocument();
    const button = await screen.findByTestId("add-secret-button");
    await userEvent.click(button);

    const secretForm = screen.getByTestId("add-secret-form");
    expect(secretForm).toBeInTheDocument();

    // enter details
    const nameInput = within(secretForm).getByTestId("name-input");
    const valueInput = within(secretForm).getByTestId("value-input");
    const submitButton = within(secretForm).getByTestId("submit-button");

    await userEvent.type(nameInput, MOCK_GET_SECRETS_RESPONSE[0].name);
    await userEvent.type(valueInput, "my-custom-secret-value");
    await userEvent.click(submitButton);

    // make POST request
    expect(createSecretSpy).not.toHaveBeenCalled();
    expect(
      screen.queryByText("SECRETS$SECRET_ALREADY_EXISTS"),
    ).toBeInTheDocument();

    expect(nameInput).toHaveValue(MOCK_GET_SECRETS_RESPONSE[0].name);
    expect(valueInput).toHaveValue("my-custom-secret-value");
  });
});
