import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SetupLlmStep } from "#/components/features/onboarding/steps/setup-llm-step";
import ConfigService from "#/api/config-service/config-service.api";
import { useFreeModelsStore } from "#/stores/free-models-store";
import SettingsService from "#/api/settings-service/settings-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { http, HttpResponse } from "msw";
import { server } from "#/mocks/node";

afterEach(() => vi.restoreAllMocks());

describe("SetupLlmStep provider selection", () => {
  it("preserves schema-backed routing and reasoning settings in the activated profile", async () => {
    const user = userEvent.setup();
    useFreeModelsStore.getState().setFlags({
      freeModels: new Set(),
      defaultModel: "openrouter/vendor/model",
    });
    const fields = [
      {
        key: "llm.litellm_extra_body",
        value_type: "object" as const,
        default: {},
      },
      {
        key: "llm.reasoning_effort",
        value_type: "string" as const,
        default: "medium",
      },
    ].map((field) => ({
      ...field,
      label: field.key,
      section: "llm",
      section_label: "LLM",
      choices: [],
      depends_on: [],
      prominence: "major" as const,
      secret: false,
      required: false,
    }));
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      agent_settings_schema: {
        ...MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
        sections:
          MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!.sections.map(
            (section) =>
              section.key === "llm"
                ? { ...section, fields: [...section.fields, ...fields] }
                : section,
          ),
      },
      agent_settings: {
        llm: {
          model: "openrouter/vendor/model",
          api_key: "test-key",
          reasoning_effort: "high",
          litellm_extra_body: {
            provider: { zdr: true, require_parameters: true },
          },
        },
      },
    });
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    const saveProfile = vi
      .spyOn(ProfilesService, "saveProfile")
      .mockResolvedValue({ name: "model", message: "Saved" });
    vi.spyOn(ProfilesService, "activateProfile").mockResolvedValue({
      name: "model",
      message: "Activated",
      llm_applied: true,
    });
    server.use(
      http.get("https://openrouter.ai/api/v1/models", () =>
        HttpResponse.json({ data: [{ id: "vendor/model" }] }),
      ),
    );
    render(
      <MemoryRouter>
        <QueryClientProvider
          client={
            new QueryClient({ defaultOptions: { queries: { retry: false } } })
          }
        >
          <SetupLlmStep onBack={() => {}} onNext={() => {}} />
        </QueryClientProvider>
      </MemoryRouter>,
    );
    await screen.findByTestId("llm-settings-screen");
    await user.click(screen.getByTestId("onboarding-llm-next"));
    await waitFor(() =>
      expect(saveProfile).toHaveBeenCalledWith(
        "model",
        expect.objectContaining({
          llm: expect.objectContaining({
            model: "openrouter/vendor/model",
            reasoning_effort: "high",
            litellm_extra_body: {
              provider: { zdr: true, require_parameters: true },
            },
          }),
        }),
      ),
    );
  });
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
