import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ProviderConnectionModal } from "#/components/features/settings/llm-profiles/provider-connection-modal";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        SETTINGS$PROVIDER_CONNECTION_ADD_TITLE: "Add provider connection",
        SETTINGS$NAME: "Name",
        SETTINGS$PROVIDER_CONNECTION_PROVIDER: "Provider",
        SETTINGS_FORM$API_KEY: "API Key",
        SETTINGS$BASE_URL: "Base URL",
        BUTTON$SAVE: "Save",
        BUTTON$CANCEL: "Cancel",
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock("#/hooks/query/use-search-providers", () => ({
  useSearchProviders: () => ({
    data: [{ name: "openai", verified: true }],
  }),
}));

vi.mock("#/hooks/mutation/use-create-provider-connection", () => ({
  useCreateProviderConnection: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
}));

vi.mock("#/hooks/mutation/use-update-provider-connection", () => ({
  useUpdateProviderConnection: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
}));

describe("ProviderConnectionModal - presets", () => {
  let queryClient: QueryClient;

  const renderModal = () =>
    render(
      <QueryClientProvider client={queryClient}>
        <ProviderConnectionModal isCreate connection={null} onClose={vi.fn()} />
      </QueryClientProvider>,
    );

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  it("shows a preset chip for Ollama Cloud", () => {
    renderModal();
    expect(
      screen.getByTestId("provider-connection-preset-ollama-cloud"),
    ).toBeInTheDocument();
  });

  it("fills the base URL and provider when the Ollama Cloud preset is clicked", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(
      screen.getByTestId("provider-connection-preset-ollama-cloud"),
    );

    expect(
      screen.getByTestId("provider-connection-base-url-input"),
    ).toHaveValue("https://ollama.com/v1");
    expect(
      screen.getByTestId("provider-connection-provider-input"),
    ).toHaveValue("OpenAI");
  });

  it("prefills the name from the preset when the name field is still empty", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(
      screen.getByTestId("provider-connection-preset-ollama-cloud"),
    );

    expect(screen.getByTestId("provider-connection-name-input")).toHaveValue(
      "Ollama Cloud",
    );
  });

  it("does not overwrite a name the user already typed", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(
      screen.getByTestId("provider-connection-name-input"),
      "My custom name",
    );
    await user.click(
      screen.getByTestId("provider-connection-preset-ollama-cloud"),
    );

    expect(screen.getByTestId("provider-connection-name-input")).toHaveValue(
      "My custom name",
    );
  });
});
