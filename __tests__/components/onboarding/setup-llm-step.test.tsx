import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SetupLlmStep } from "#/components/features/onboarding/steps/setup-llm-step";
import ConfigService from "#/api/config-service/config-service.api";
import { useFreeModelsStore } from "#/stores/free-models-store";

afterEach(() => vi.restoreAllMocks());

describe("SetupLlmStep provider selection", () => {
  it("blocks Next until a model is selected after changing provider", async () => {
    const user = userEvent.setup();
    useFreeModelsStore
      .getState()
      .setFlags({ freeModels: new Set(), defaultModel: "openai/gpt-4o" });
    vi.spyOn(ConfigService, "searchProviders").mockResolvedValue({
      items: [
        { name: "openai", verified: true },
        { name: "anthropic", verified: true },
      ],
      next_page_id: null,
    });
    const onNext = vi.fn();
    render(
      <MemoryRouter>
        <QueryClientProvider
          client={
            new QueryClient({ defaultOptions: { queries: { retry: false } } })
          }
        >
          <SetupLlmStep onBack={() => {}} onNext={onNext} />
        </QueryClientProvider>
      </MemoryRouter>,
    );
    const provider = await screen.findByTestId("llm-provider-input");
    await user.click(provider);
    await user.click(await screen.findByText("Anthropic"));
    await waitFor(() =>
      expect(screen.getByTestId("onboarding-llm-next")).toBeDisabled(),
    );
    expect(onNext).not.toHaveBeenCalled();
  });
});
