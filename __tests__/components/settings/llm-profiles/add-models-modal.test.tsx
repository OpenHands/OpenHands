import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AddModelsModal } from "#/components/features/settings/llm-profiles/add-models-modal";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import ConfigService from "#/api/config-service/config-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        SETTINGS$ADD_MODELS_TITLE: "Add models as profiles",
        SETTINGS$ADD_MODELS_PROVIDER_LABEL: "Provider",
        SETTINGS$ADD_MODELS_PROVIDER_PLACEHOLDER: "Select a provider",
        SETTINGS$ADD_MODELS_VERIFIED_ONLY: "Verified models only",
        SETTINGS$ADD_MODELS_SELECT_ALL: "Select all",
        SETTINGS$ADD_MODELS_EMPTY: "No models found for this provider.",
        SETTINGS$ADD_MODELS_NAME_TAKEN: "Name already exists",
        SETTINGS$ADD_N_PROFILES: `Add ${params?.count ?? "?"} profiles`,
        SETTINGS$MODELS_ADDED: `Added ${params?.count ?? "?"} profiles`,
        SETTINGS$MODELS_ADDED_PARTIAL: `Added ${params?.added ?? "?"} profiles; ${params?.failed ?? "?"} failed`,
        SETTINGS$MODEL_ROW_SAVED: "Saved",
        SETTINGS$MODEL_ROW_FAILED: "Failed",
        BUTTON$CANCEL: "Cancel",
        ERROR$GENERIC: "An error occurred",
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock("#/api/profiles-service/profiles-service.api");
vi.mock("#/api/config-service/config-service.api");

// The provider/model hooks hydrate a verified-models map through the active
// backend's LLMMetadataClient; in tests there is no backend, so stub the map.
vi.mock("#/hooks/query/use-verified-models", async (importOriginal) => {
  const orig =
    await importOriginal<typeof import("#/hooks/query/use-verified-models")>();
  return {
    ...orig,
    fetchVerifiedModelsByProvider: vi.fn().mockResolvedValue({}),
  };
});

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

const PROVIDERS = {
  items: [
    { name: "openhands", verified: true },
    { name: "openai", verified: true },
  ],
  next_page_id: null,
};

const MODELS = {
  items: [
    { provider: "openhands", name: "trinity-large-thinking", verified: true },
    { provider: "openhands", name: "deepseek-v4-flash", verified: true },
    { provider: "openhands", name: "unverified-model", verified: false },
  ],
  next_page_id: null,
};

describe("AddModelsModal", () => {
  let queryClient: QueryClient;

  const renderModal = (existingNames: string[] = [], onClose = vi.fn()) =>
    render(
      <QueryClientProvider client={queryClient}>
        <AddModelsModal
          isOpen
          existingNames={existingNames}
          onClose={onClose}
        />
      </QueryClientProvider>,
    );

  const pickProvider = async () => {
    const select = await screen.findByTestId("add-models-provider");
    // Options populate async via the providers query; wait before selecting.
    await screen.findByRole("option", { name: "openhands" });
    await userEvent.selectOptions(select, "openhands");
  };

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    vi.mocked(ConfigService.searchProviders).mockResolvedValue(PROVIDERS);
    vi.mocked(ConfigService.searchModels).mockResolvedValue(MODELS);
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue({
      name: "x",
      message: "ok",
    });
  });

  it("lists verified models with derived names after picking a provider", async () => {
    renderModal();
    await pickProvider();

    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );
    expect(
      screen.getByTestId("add-models-row-openhands/deepseek-v4-flash"),
    ).toBeInTheDocument();
    // unverified filtered out by default
    expect(
      screen.queryByTestId("add-models-row-openhands/unverified-model"),
    ).not.toBeInTheDocument();
    // derived name pre-fills the input
    expect(
      screen.getByTestId("add-models-name-openhands/deepseek-v4-flash"),
    ).toHaveValue("deepseek-v4-flash");
  });

  it("shows unverified models when the filter is toggled off", async () => {
    renderModal();
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );

    await userEvent.click(screen.getByTestId("add-models-verified-only"));

    await screen.findByTestId("add-models-row-openhands/unverified-model");
  });

  it("flags names that collide with existing profiles and excludes them", async () => {
    renderModal(["deepseek-v4-flash"]);
    await pickProvider();
    await screen.findByTestId("add-models-row-openhands/deepseek-v4-flash");

    expect(
      screen.getByTestId("add-models-conflict-openhands/deepseek-v4-flash"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("add-models-check-openhands/deepseek-v4-flash"),
    ).toBeDisabled();
    // only the non-colliding model is offered for submit
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 1 profiles",
    );
  });

  it("creates one profile per selected model, keyless", async () => {
    const onClose = vi.fn();
    renderModal([], onClose);
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );

    // deselect one; only the other is created
    await userEvent.click(
      screen.getByTestId("add-models-check-openhands/trinity-large-thinking"),
    );
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(ProfilesService.saveProfile).toHaveBeenCalledTimes(1);
    expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
      "deepseek-v4-flash",
      {
        llm: { model: "openhands/deepseek-v4-flash" },
        include_secrets: false,
      },
    );
  });

  it("keeps the modal open and marks the row on per-model failure", async () => {
    const onClose = vi.fn();
    vi.mocked(ProfilesService.saveProfile)
      .mockResolvedValueOnce({ name: "a", message: "ok" })
      .mockRejectedValueOnce(new Error("boom"));
    renderModal([], onClose);
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );

    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );
    expect(onClose).not.toHaveBeenCalled();
    const rows = screen.getAllByText(/^(Saved|Failed)$/);
    expect(rows).toHaveLength(2);
  });

  it("select-all toggles every selectable row", async () => {
    renderModal();
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 0 profiles",
    );
    expect(screen.getByTestId("add-models-submit")).toBeDisabled();

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 2 profiles",
    );
  });
});
