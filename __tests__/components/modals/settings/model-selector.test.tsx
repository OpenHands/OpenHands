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

  it("selects provider and model with the keyboard", async () => {
    // Arrange
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithQuery(<ModelSelector onChange={onChange} />);

    // Act: filter to and commit the Azure provider with Enter.
    const providerSelector = screen.getByLabelText("LLM Provider");
    await user.click(providerSelector);
    await user.clear(providerSelector);
    await user.keyboard("azur{ArrowDown}{Enter}");

    // Assert
    expect(onChange).toHaveBeenNthCalledWith(1, "azure", null);
    expect(providerSelector).toHaveValue("Azure");

    // Act: commit the first Azure model with the keyboard.
    const modelSelector = screen.getByLabelText("LLM Model");
    await user.click(modelSelector);
    await user.keyboard("{ArrowDown}{Enter}");

    // Assert
    expect(onChange).toHaveBeenNthCalledWith(2, "azure", "ada");
    expect(modelSelector).toHaveValue("ada");
  });

  it("reverts a typed provider name that is not committed on blur", async () => {
    // Arrange
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithQuery(
      <div>
        <ModelSelector currentModel="azure/ada" onChange={onChange} />
        <button type="button">Outside</button>
      </div>,
    );
    const providerSelector = screen.getByLabelText("LLM Provider");
    await waitFor(() => expect(providerSelector).toHaveValue("Azure"));

    // Act: type a partial provider name and blur without committing it.
    await user.click(providerSelector);
    await user.keyboard("open");
    await user.click(screen.getByText("Outside"));

    // Assert: neither the provider nor the model selection changes.
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByLabelText("LLM Model")).toHaveValue("ada");
  });

  it("clears provider and model when the provider input is emptied", async () => {
    // Arrange
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithQuery(
      <ModelSelector currentModel="azure/ada" onChange={onChange} />,
    );
    const providerSelector = screen.getByLabelText("LLM Provider");
    await waitFor(() => expect(providerSelector).toHaveValue("Azure"));
    onChange.mockClear();

    // Act
    await user.click(providerSelector);
    await user.clear(providerSelector);

    // Assert: the model input is disabled and emptied with the provider.
    const modelSelector = screen.getByLabelText("LLM Model");
    expect(modelSelector).toBeDisabled();
    expect(modelSelector).toHaveValue("");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("renders a single field border on provider and model comboboxes", () => {
    // Arrange / Act
    renderWithQuery(<ModelSelector />);

    // Assert
    for (const testId of ["llm-provider-input", "llm-model-input"]) {
      const input = screen.getByTestId(testId);
      expect(input.className).toContain("border-0");
      expect(input.className).toContain("bg-transparent");
      expect(input.className).toContain("focus:ring-0");
      const group = input.closest('[data-slot="combo-box-input-group"]');
      expect(group?.className).toContain("border-control");
      expect(group?.className).toContain("focus-within:ring-1");
    }
  });
});
