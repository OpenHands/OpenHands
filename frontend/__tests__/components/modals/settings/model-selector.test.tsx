import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ModelSelector } from "#/components/shared/modals/settings/model-selector";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: { [key: string]: string } = {
        LLM$PROVIDER: "LLM Provider",
        LLM$MODEL: "LLM Model",
        LLM$SELECT_PROVIDER_PLACEHOLDER: "Select a provider",
        LLM$SELECT_MODEL_PLACEHOLDER: "Select a model",
      };
      return translations[key] || key;
    },
  }),
}));

describe("ModelSelector", () => {
  const models = {
    openai: {
      separator: "/",
      models: ["gpt-4o", "gpt-4o-mini"],
    },
    azure: {
      separator: "/",
      models: ["ada", "gpt-35-turbo"],
    },
    vertex_ai: {
      separator: "/",
      models: ["chat-bison", "chat-bison-32k"],
    },
  };

  const verifiedModels = ["gpt-4o", "gpt-4o-mini"];
  const verifiedProviders = ["openai"];

  it("should display the provider selector", async () => {
    const user = userEvent.setup();
    render(
      <ModelSelector
        models={models}
        verifiedModels={verifiedModels}
        verifiedProviders={verifiedProviders}
      />,
    );

    const selector = screen.getByLabelText("LLM Provider");
    expect(selector).toBeInTheDocument();

    await user.click(selector);

    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByText("Azure")).toBeInTheDocument();
    expect(screen.getByText("VertexAI")).toBeInTheDocument();
  });

  it("should disable the model selector if the provider is not selected", async () => {
    const user = userEvent.setup();
    render(
      <ModelSelector
        models={models}
        verifiedModels={verifiedModels}
        verifiedProviders={verifiedProviders}
      />,
    );

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
    render(
      <ModelSelector
        models={models}
        verifiedModels={verifiedModels}
        verifiedProviders={verifiedProviders}
      />,
    );

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

    render(
      <ModelSelector
        models={models}
        verifiedModels={verifiedModels}
        verifiedProviders={verifiedProviders}
        onChange={onChange}
      />,
    );

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
    render(
      <ModelSelector
        models={models}
        verifiedModels={verifiedModels}
        verifiedProviders={verifiedProviders}
        currentModel="azure/ada"
      />,
    );

    expect(screen.getByLabelText("LLM Provider")).toHaveValue("Azure");
    expect(screen.getByLabelText("LLM Model")).toHaveValue("ada");
  });
});
