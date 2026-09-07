import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SecretsSettingsScreen } from "#/routes/secrets-settings";
import { SecretsService } from "#/api/secrets-service";

function renderSecretsSettingsScreen() {
  return render(<SecretsSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={new QueryClient({
          defaultOptions: { queries: { retry: false } },
        })}
      >
        {children}
      </QueryClientProvider>
    ),
  });
}

describe("SecretsSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the OSS secrets list for local secrets management", async () => {
    // Mock getSecrets (used by useSearchSecrets internally)
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([
      {
        name: "MY_SECRET",
        description: "Demo secret",
      },
    ]);

    renderSecretsSettingsScreen();

    await screen.findByTestId("secrets-settings-screen");
    expect(await screen.findByText("MY_SECRET")).toBeInTheDocument();
    expect(screen.getByTestId("add-secret-button")).toBeInTheDocument();
  });

  it("disables Add Secret submit until required fields are filled", async () => {
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
    const user = userEvent.setup();

    renderSecretsSettingsScreen();

    await screen.findByTestId("secrets-settings-screen");
    await user.click(screen.getByTestId("add-secret-button"));

    const submitButton = screen.getByTestId("submit-button");
    const nameInput = screen.getByTestId("name-input");
    const valueInput = screen.getByTestId("value-input");

    expect(submitButton).toBeDisabled();

    await user.type(nameInput, "MY_SECRET");
    expect(submitButton).toBeDisabled();

    await user.type(valueInput, "super-secret");
    expect(submitButton).not.toBeDisabled();
  });

  it("disables Edit Secret submit until a field changes", async () => {
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([
      {
        name: "MY_SECRET",
        description: "Demo secret",
      },
    ]);
    const user = userEvent.setup();

    renderSecretsSettingsScreen();

    await screen.findByTestId("secrets-settings-screen");
    await screen.findByText("MY_SECRET");
    await user.click(screen.getByTestId("edit-secret-button"));

    const submitButton = screen.getByTestId("submit-button");
    const descriptionInput = screen.getByTestId("description-input");

    expect(submitButton).toBeDisabled();

    await user.type(descriptionInput, " updated");
    expect(submitButton).not.toBeDisabled();
  });
});