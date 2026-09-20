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
  { name: "openrouter", verified: false },
];

const model = (partial: Partial<LLMModel> & Pick<LLMModel, "name">): LLMModel => ({
  provider: null,
  verified: false,
  free: false,
  default: false,
  ...partial,
});

const mockModelsByProvider: Record<string, LLMModel[]> = {
  openai: [
    model({ provider: "openai", name: "gpt-4o", verified: true }),
    model({ provider: "openai", name: "gpt-4o-mini", verified: true }),
  ],
  azure: [
    model({ provider: "azure", name: "ada" }),
    model({ provider: "azure", name: "gpt-35-turbo" }),
  ],
  vertex_ai: [
    model({ provider: "vertex_ai", name: "chat-bison" }),
    model({ provider: "vertex_ai", name: "chat-bison-32k" }),
  ],
  openrouter: [model({ provider: "openrouter", name: "anthropic/claude-3.5" })],
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
        SETTINGS$CUSTOM_MODEL: "Custom Model",
        MODEL_SELECTOR$CUSTOM_MODEL_HINT:
          "Enter the model ID as the provider lists it.",
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

  it("should not render placeholder text on the provider or model inputs", () => {
    renderWithQuery(<ModelSelector />);

    const providerInput = screen.getByLabelText("LLM Provider");
    const modelInput = screen.getByLabelText("LLM Model");

    expect(providerInput.getAttribute("placeholder") ?? "").toBe("");
    expect(modelInput.getAttribute("placeholder") ?? "").toBe("");
  });

  it("should report a model the provider catalog does not list", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    renderWithQuery(<ModelSelector onChange={onChange} />);

    await user.click(screen.getByLabelText("LLM Provider"));
    await user.click(screen.getByText("OpenRouter"));

    await user.click(screen.getByLabelText("LLM Model"));
    await user.click(screen.getByTestId("model-item-custom"));

    const customInput = await screen.findByTestId("custom-model-input");
    // `SettingsForm` submits the raw form, so the visible field must be the one
    // carrying the model's form name.
    expect(customInput).toHaveAttribute("name", "llm-model-input");
    await user.type(customInput, "thinkingmachines/inkling-small:free");

    expect(onChange).toHaveBeenLastCalledWith(
      "openrouter",
      "thinkingmachines/inkling-small:free",
    );
  });

  it("should edit a saved model the provider catalog does not list", async () => {
    renderWithQuery(
      <ModelSelector currentModel="openrouter/thinkingmachines/inkling-small:free" />,
    );

    await waitFor(() => {
      expect(screen.getByLabelText("LLM Provider")).toHaveValue("OpenRouter");
    });

    expect(await screen.findByTestId("custom-model-input")).toHaveValue(
      "thinkingmachines/inkling-small:free",
    );
  });

  it("should not offer the custom model input for a listed model", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ModelSelector />);

    await user.click(screen.getByLabelText("LLM Provider"));
    await user.click(screen.getByText("Azure"));

    await user.click(screen.getByLabelText("LLM Model"));
    await user.click(screen.getByText("ada"));

    expect(screen.queryByTestId("custom-model-input")).not.toBeInTheDocument();
  });
});
