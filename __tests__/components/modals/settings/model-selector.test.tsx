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
  { name: "openrouter", verified: true },
];

const model = (
  partial: Partial<LLMModel> & Pick<LLMModel, "name">,
): LLMModel => ({
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
  openrouter: [
    model({
      provider: "openrouter",
      name: "vendor/tool-model",
      capabilities: {
        supportedParameters: ["tools", "reasoning"],
        contextLength: 128000,
        maxOutputTokens: 8192,
        pricing: { prompt: "0.000003", completion: "0.000015" },
      },
    }),
    model({
      provider: "openrouter",
      name: "vendor/no-tools-model",
      capabilities: { supportedParameters: ["temperature"] },
    }),
    model({
      provider: "openrouter",
      name: "vendor/unknown-capability-model",
    }),
    model({
      provider: "openrouter",
      name: "vendor/auto-priced-model",
      capabilities: {
        supportedParameters: ["tool_choice", "reasoning_effort"],
        pricing: { prompt: "-1", completion: "-1" },
      },
    }),
    model({
      provider: "openrouter",
      name: "vendor/blank-priced-model",
      capabilities: {
        supportedParameters: ["tools"],
        pricing: { prompt: "   ", completion: "0.000003" },
      },
    }),
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
    t: (key: string, values?: Record<string, string>) => {
      const translations: Record<string, string> = {
        LLM$PROVIDER: "LLM Provider",
        LLM$MODEL: "LLM Model",
        LLM$SELECT_PROVIDER_PLACEHOLDER: "Select a provider",
        LLM$SELECT_MODEL_PLACEHOLDER: "Select a model",
        MODEL_SELECTOR$SHOW_ALL_OPENROUTER_MODELS: "Show all models",
        MODEL_SELECTOR$CONTEXT_LENGTH: "Context: {{value}} tokens",
        MODEL_SELECTOR$MAX_OUTPUT: "Max output: {{value}} tokens",
        MODEL_SELECTOR$SUPPORTS_TOOLS: "Tools",
        MODEL_SELECTOR$SUPPORTS_REASONING: "Reasoning",
        MODEL_SELECTOR$PRICING: "{{prompt}}/M in · {{completion}}/M out",
      };
      const template = translations[key] || key;
      if (!values) return template;
      return Object.entries(values).reduce(
        (text, [name, value]) => text.replaceAll(`{{${name}}}`, value),
        template,
      );
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
});

describe("ModelSelector — fixedProvider", () => {
  it("disables the provider input and fetches the fixed provider's models even with an empty current model", async () => {
    renderWithQuery(<ModelSelector fixedProvider="azure" />);

    const providerInput = screen.getByLabelText("LLM Provider");
    await waitFor(() => expect(providerInput).toHaveValue("Azure"));
    expect(providerInput).toBeDisabled();

    const user = userEvent.setup();
    const modelSelector = screen.getByLabelText("LLM Model");
    expect(modelSelector).not.toBeDisabled();
    await user.click(modelSelector);
    expect(screen.getByText("ada")).toBeInTheDocument();
  });

  it("does not reset the model when re-rendered with the same fixed provider", async () => {
    const { rerender } = renderWithQuery(
      <ModelSelector fixedProvider="azure" currentModel="azure/ada" />,
    );
    await waitFor(() =>
      expect(screen.getByLabelText("LLM Model")).toHaveValue("ada"),
    );

    rerender(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ModelSelector fixedProvider="azure" currentModel="azure/ada" />
      </QueryClientProvider>,
    );

    expect(screen.getByLabelText("LLM Model")).toHaveValue("ada");
  });

  it("clears a stale model when the caller empties currentModel under the same fixed provider (incompatible connection swap)", async () => {
    const { rerender } = renderWithQuery(
      <ModelSelector fixedProvider="azure" currentModel="azure/ada" />,
    );
    await waitFor(() =>
      expect(screen.getByLabelText("LLM Model")).toHaveValue("ada"),
    );

    rerender(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ModelSelector fixedProvider="azure" currentModel="" />
      </QueryClientProvider>,
    );

    await waitFor(() =>
      expect(screen.getByLabelText("LLM Model")).toHaveValue(""),
    );
    expect(screen.getByLabelText("LLM Provider")).toHaveValue("Azure");
  });
});

describe("ModelSelector — OpenRouter capability-aware filtering", () => {
  it("defaults to hiding known tool-incapable models while keeping tool-capable and unknown-capability ones", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ModelSelector fixedProvider="openrouter" />);

    await user.click(screen.getByLabelText("LLM Model"));

    expect(screen.getByText("vendor/tool-model")).toBeInTheDocument();
    expect(
      screen.getByText("vendor/unknown-capability-model"),
    ).toBeInTheDocument();
    expect(screen.queryByText("vendor/no-tools-model")).not.toBeInTheDocument();
  });

  it("reveals every model once the show-all escape hatch is checked", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ModelSelector fixedProvider="openrouter" />);

    const showAllToggle = screen.getByTestId(
      "openrouter-show-all-models-toggle",
    );
    await user.click(showAllToggle);
    await user.click(screen.getByLabelText("LLM Model"));

    expect(screen.getByText("vendor/no-tools-model")).toBeInTheDocument();
  });

  it("never hides the already-selected model, even if it is known tool-incapable", async () => {
    renderWithQuery(
      <ModelSelector currentModel="openrouter/vendor/no-tools-model" />,
    );

    await waitFor(() =>
      expect(screen.getByLabelText("LLM Model")).toHaveValue(
        "vendor/no-tools-model",
      ),
    );
  });

  it("does not render the show-all toggle for non-OpenRouter providers", () => {
    renderWithQuery(<ModelSelector fixedProvider="azure" />);

    expect(
      screen.queryByTestId("openrouter-show-all-models-toggle"),
    ).not.toBeInTheDocument();
  });
});

describe("ModelSelector — selected model capability metadata", () => {
  it("shows context length, output cap, tools and reasoning support for the selected model", async () => {
    renderWithQuery(
      <ModelSelector currentModel="openrouter/vendor/tool-model" />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("model-capabilities")).toBeInTheDocument(),
    );
    const capabilities = screen.getByTestId("model-capabilities");
    expect(capabilities).toHaveTextContent("128k");
    expect(capabilities).toHaveTextContent("8.2k");
    expect(screen.getByTestId("model-supports-tools")).toBeInTheDocument();
    expect(screen.getByTestId("model-supports-reasoning")).toBeInTheDocument();
  });

  it("renders no capability metadata for a model without known capabilities", async () => {
    renderWithQuery(<ModelSelector currentModel="azure/ada" />);

    await waitFor(() =>
      expect(screen.getByLabelText("LLM Model")).toHaveValue("ada"),
    );
    expect(screen.queryByTestId("model-capabilities")).not.toBeInTheDocument();
  });

  it("never renders a price for OpenRouter's -1 'varies' sentinel", async () => {
    renderWithQuery(
      <ModelSelector currentModel="openrouter/vendor/auto-priced-model" />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("model-capabilities")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("model-pricing")).not.toBeInTheDocument();
  });

  it("never renders a price built from a whitespace-only price string", async () => {
    renderWithQuery(
      <ModelSelector currentModel="openrouter/vendor/blank-priced-model" />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("model-capabilities")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("model-pricing")).not.toBeInTheDocument();
  });

  it("requires 'tools' specifically — 'tool_choice' alone does not count as tool support", async () => {
    renderWithQuery(
      <ModelSelector currentModel="openrouter/vendor/auto-priced-model" />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("model-capabilities")).toBeInTheDocument(),
    );
    expect(
      screen.queryByTestId("model-supports-tools"),
    ).not.toBeInTheDocument();
  });

  it("treats 'reasoning_effort' as reasoning support", async () => {
    renderWithQuery(
      <ModelSelector currentModel="openrouter/vendor/auto-priced-model" />,
    );

    await waitFor(() =>
      expect(
        screen.getByTestId("model-supports-reasoning"),
      ).toBeInTheDocument(),
    );
  });
});
