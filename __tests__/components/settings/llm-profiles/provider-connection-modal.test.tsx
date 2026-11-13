import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ProviderConnectionModal } from "#/components/features/settings/llm-profiles/provider-connection-modal";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("#/hooks/query/use-search-providers", () => ({
  useSearchProviders: () => ({
    data: [
      { name: "openai", verified: true },
      { name: "anthropic", verified: true },
      { name: "my-custom-provider", verified: false },
    ],
  }),
}));

const createMutate = vi.hoisted(() => vi.fn());
const updateMutate = vi.hoisted(() => vi.fn());

vi.mock("#/hooks/mutation/use-create-provider-connection", () => ({
  useCreateProviderConnection: () => ({
    isPending: false,
    mutateAsync: createMutate,
  }),
}));

vi.mock("#/hooks/mutation/use-update-provider-connection", () => ({
  useUpdateProviderConnection: () => ({
    isPending: false,
    mutateAsync: updateMutate,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

function renderWithQuery(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
  );
}

const existingConnection: ProviderConnection = {
  id: "conn-1",
  display_name: "My connection",
  provider: "my-custom-provider",
  base_url: null,
  created_at: 0,
  updated_at: 0,
  api_key_set: true,
};

describe("ProviderConnectionModal provider combobox", () => {
  it("shows the provider combobox in edit mode with a provider missing from the list", async () => {
    // Arrange
    renderWithQuery(
      <ProviderConnectionModal
        connection={existingConnection}
        isCreate={false}
        onClose={vi.fn()}
      />,
    );

    // Assert: main keeps the combobox (not a plain text input) in edit mode,
    // and a provider absent from the fetched list still renders selected.
    const providerInput = screen.getByTestId(
      "provider-connection-provider-input",
    );
    expect(providerInput).toHaveAttribute("role", "combobox");
    await waitFor(() =>
      expect(providerInput).toHaveValue("my-custom-provider"),
    );
  });

  it("selects a provider with the keyboard in create mode", async () => {
    // Arrange
    const user = userEvent.setup();
    renderWithQuery(<ProviderConnectionModal isCreate onClose={vi.fn()} />);

    // Act
    const providerInput = screen.getByTestId(
      "provider-connection-provider-input",
    );
    await user.click(providerInput);
    await user.keyboard("anth{ArrowDown}{Enter}");

    // Assert
    await waitFor(() => expect(providerInput).toHaveValue("Anthropic"));
  });

  it("reverts an uncommitted provider name on blur in create mode", async () => {
    // Arrange
    const user = userEvent.setup();
    renderWithQuery(
      <div>
        <ProviderConnectionModal isCreate onClose={vi.fn()} />
        <button type="button">Outside</button>
      </div>,
    );
    const providerInput = screen.getByTestId(
      "provider-connection-provider-input",
    );
    await user.click(providerInput);
    await user.keyboard("anth{ArrowDown}{Enter}");
    await waitFor(() => expect(providerInput).toHaveValue("Anthropic"));

    // Act: type over the selected label and blur without committing.
    await user.click(providerInput);
    await user.keyboard("open");
    await user.click(screen.getByText("Outside"));

    // Assert: the committed provider is kept.
    await waitFor(() => expect(providerInput).toHaveValue("Anthropic"));
  });

  it("renders a single field border on the provider combobox", () => {
    // Arrange / Act
    renderWithQuery(<ProviderConnectionModal isCreate onClose={vi.fn()} />);

    // Assert
    const providerInput = screen.getByTestId(
      "provider-connection-provider-input",
    );
    expect(providerInput.className).toContain("border-0");
    expect(providerInput.className).toContain("bg-transparent");
    expect(providerInput.className).toContain("focus:ring-0");
    const group = providerInput.closest('[data-slot="combo-box-input-group"]');
    expect(group?.className).toContain("border-control");
    expect(group?.className).toContain("focus-within:ring-1");
  });
});
