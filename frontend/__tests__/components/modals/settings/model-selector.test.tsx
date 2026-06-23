import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ModelSelector } from "#/components/shared/modals/settings/model-selector";
import type {
  LLMProvider,
  LLMModel,
} from "#/api/config-service/config-service.types";

const mockProviders: LLMProvider[] = [
  { name: "openai", verified: true },
  { name: "azure", verified: false },
  { name: "vertex_ai", verified: false },
  // Provider whose model list comes back empty (no entry in
  // mockModelsByProvider) — used by the unavailable-model warning tests.
  { name: "groq", verified: false },
];

const mockModelsByProvider: Record<string, LLMModel[]> = {
  openai: [
    { provider: "openai", name: "gpt-4o", verified: true },
    { provider: "openai", name: "gpt-4o-mini", verified: true },
  ],
  azure: [
    { provider: "azure", name: "ada", verified: false },
    { provider: "azure", name: "gpt-35-turbo", verified: false },
    // Served-but-not-promoted alias route (e.g. a legacy name a managed
    // proxy keeps serving after a rename): never a dropdown option, but a
    // saved setting referencing it still counts as available.
    { provider: "azure", name: "legacy-alias", verified: false, hidden: true },
  ],
  vertex_ai: [
    { provider: "vertex_ai", name: "chat-bison", verified: false },
    { provider: "vertex_ai", name: "chat-bison-32k", verified: false },
  ],
};

vi.mock("#/hooks/query/use-search-providers", () => ({
  useSearchProviders: () => ({ data: mockProviders }),
}));

vi.mock("#/hooks/query/use-provider-models", () => ({
  useProviderModels: (provider: string | null) => ({
    data: provider ? (mockModelsByProvider[provider] ?? []) : [],
  }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        LLM$PROVIDER: "LLM Provider",
        LLM$MODEL: "LLM Model",
        LLM$SELECT_PROVIDER_PLACEHOLDER: "Select a provider",
        LLM$SELECT_MODEL_PLACEHOLDER: "Select a model",
      };
      return translations[key] || key;
    },
  }),
}));

function renderWithQuery(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
  );
}

describe("ModelSelector", () => {
  it("should display the provider selector", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ModelSelector />);

    const selector = screen.getByLabelText("LLM Provider");
    expect(selector).toBeInTheDocument();

    await user.click(selector);

    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByText("Azure")).toBeInTheDocument();
    expect(screen.getByText("VertexAI")).toBeInTheDocument();
  });

  it("should disable the model selector if the provider is not selected", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ModelSelector />);

    const modelSelector = screen.getByLabelText("LLM Model");
    expect(modelSelector).toBeDisabled();

    const providerSelector = screen.getByLabelText("LLM Provider");
    await user.click(providerSelector);

    const vertexAI = screen.getByText("VertexAI");
    await user.click(vertexAI);

    expect(modelSelector).not.toBeDisabled();
  });

  it("should display the model selector", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ModelSelector />);

    const providerSelector = screen.getByLabelText("LLM Provider");
    await user.click(providerSelector);

    const azureProvider = screen.getByText("Azure");
    await user.click(azureProvider);

    const modelSelector = screen.getByLabelText("LLM Model");
    await user.click(modelSelector);

    expect(screen.getByText("ada")).toBeInTheDocument();
    expect(screen.getByText("gpt-35-turbo")).toBeInTheDocument();
    // Hidden alias routes are served by the backend but never promoted.
    expect(screen.queryByText("legacy-alias")).not.toBeInTheDocument();
  });

  it("should call onChange when the provider and model change", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    renderWithQuery(<ModelSelector onChange={onChange} />);

    const providerSelector = screen.getByLabelText("LLM Provider");
    await user.click(providerSelector);
    await user.click(screen.getByText("Azure"));

    const modelSelector = screen.getByLabelText("LLM Model");
    await user.click(modelSelector);
    await user.click(screen.getByText("ada"));

    expect(onChange).toHaveBeenNthCalledWith(1, "azure", null);
    expect(onChange).toHaveBeenNthCalledWith(2, "azure", "ada");
  });

  it("should have a default value if passed", async () => {
    renderWithQuery(<ModelSelector currentModel="azure/ada" />);

    await waitFor(() => {
      expect(screen.getByLabelText("LLM Provider")).toHaveValue("Azure");
      expect(screen.getByLabelText("LLM Model")).toHaveValue("ada");
    });
  });

  it("should stretch the provider and model selectors across the full row", () => {
    const { container } = renderWithQuery(<ModelSelector />);

    expect(container.firstChild).toHaveClass("w-full");
  });

  describe("unavailable model warning", () => {
    it("shows a warning when the saved model is missing from a non-empty model list", async () => {
      renderWithQuery(<ModelSelector currentModel="azure/removed-model" />);

      await waitFor(() => {
        expect(
          screen.getByTestId("model-unavailable-warning"),
        ).toBeInTheDocument();
      });
    });

    it("does not show a warning when the saved model matches a hidden alias", async () => {
      // "legacy-alias" is served by the backend (hidden=true) but is not a
      // dropdown option — the saved setting still routes, so no warning.
      renderWithQuery(<ModelSelector currentModel="azure/legacy-alias" />);

      await waitFor(() => {
        expect(screen.getByLabelText("LLM Provider")).toHaveValue("Azure");
      });
      expect(
        screen.queryByTestId("model-unavailable-warning"),
      ).not.toBeInTheDocument();
    });

    it("does not show a warning when the saved model is in the model list", async () => {
      renderWithQuery(<ModelSelector currentModel="azure/ada" />);

      await waitFor(() => {
        expect(screen.getByLabelText("LLM Model")).toHaveValue("ada");
      });
      expect(
        screen.queryByTestId("model-unavailable-warning"),
      ).not.toBeInTheDocument();
    });

    it("does not show a warning when the provider's model list is empty", async () => {
      // "groq" has no entry in mockModelsByProvider, so the hook returns []
      // — an empty list must not cast doubt on the saved model.
      renderWithQuery(<ModelSelector currentModel="groq/llama3-70b" />);

      await waitFor(() => {
        expect(screen.getByLabelText("LLM Provider")).toHaveValue("Groq");
      });
      expect(
        screen.queryByTestId("model-unavailable-warning"),
      ).not.toBeInTheDocument();
    });

    it("clears the warning when the user picks an available model", async () => {
      const user = userEvent.setup();
      renderWithQuery(<ModelSelector currentModel="azure/removed-model" />);

      await waitFor(() => {
        expect(
          screen.getByTestId("model-unavailable-warning"),
        ).toBeInTheDocument();
      });

      const modelSelector = screen.getByLabelText("LLM Model");
      await user.click(modelSelector);
      await user.click(screen.getByText("ada"));

      await waitFor(() => {
        expect(
          screen.queryByTestId("model-unavailable-warning"),
        ).not.toBeInTheDocument();
      });
    });
  });
});
