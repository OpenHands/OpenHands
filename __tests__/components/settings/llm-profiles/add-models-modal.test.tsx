import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AddModelsModal } from "#/components/features/settings/llm-profiles/add-models-modal";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import ConfigService from "#/api/config-service/config-service.api";
import type {
  LLMModel,
  LLMModelPage,
} from "#/api/config-service/config-service.types";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        SETTINGS$ADD_MODELS_TITLE: "Add models as profiles",
        SETTINGS$ADD_MODELS_CONNECTION_BOUND: `Linked to ${params?.provider ?? "?"}`,
        SETTINGS$ADD_MODELS_VERIFIED_ONLY: "Verified models only",
        SETTINGS$ADD_MODELS_SELECT_ALL: "Select all",
        SETTINGS$ADD_MODELS_EMPTY: "No models found for this provider.",
        COMMON$NO_RESULTS: "No results found",
        SETTINGS$ADD_MODELS_NAME_TAKEN: "Name already exists",
        SETTINGS$ADD_N_PROFILES: `Add ${params?.count ?? "?"} profiles`,
        SETTINGS$MODELS_ADDED: `Added ${params?.count ?? "?"} profiles`,
        SETTINGS$MODELS_ADDED_PARTIAL: `Added ${params?.added ?? "?"} profiles; ${params?.failed ?? "?"} failed`,
        SETTINGS$MODELS_ADD_BLOCKED: `Added ${params?.added ?? "?"} profiles; the server refused the rest`,
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

/**
 * An error shaped like the SDK's HttpError, which the modal narrows on.
 * `HttpError.response` carries the parsed error body, so a server that answers
 * with a plain-text or detail-less body leaves nothing for the modal to quote.
 */
const httpError = (status: number, detail?: string) => {
  const error = new Error(detail ?? `HTTP ${status}`);
  error.name = "HttpError";
  return Object.assign(error, {
    status,
    response: detail === undefined ? null : { detail },
  });
};

// The local reconstruction path sets `free`/`default` to false for every item;
// the fixtures mirror that so they satisfy LLMModel without drift.
const model = (
  provider: string,
  name: string,
  verified: boolean,
): LLMModel => ({ provider, name, verified, free: false, default: false });

const OPENAI_MODELS: LLMModelPage = {
  items: [
    model("openai", "gpt-4o", true),
    model("openai", "gpt-4o-mini", true),
    model("openai", "unverified-model", false),
  ],
  next_page_id: null,
};

function makeConnection(
  overrides: Partial<ProviderConnection> = {},
): ProviderConnection {
  return {
    id: "conn-openai",
    display_name: "Shared OpenAI",
    provider: "openai",
    base_url: null,
    created_at: 1,
    updated_at: 2,
    api_key_set: true,
    ...overrides,
  };
}

const OPENAI_CONNECTION = makeConnection();

describe("AddModelsModal", () => {
  let queryClient: QueryClient;

  const renderModal = (
    connection: ProviderConnection | null = OPENAI_CONNECTION,
    existingNames: string[] = [],
    onClose = vi.fn(),
  ) => {
    const view = render(
      <QueryClientProvider client={queryClient}>
        <AddModelsModal
          isOpen
          connection={connection}
          existingNames={existingNames}
          onClose={onClose}
        />
      </QueryClientProvider>,
    );
    // The manager renders the modal unconditionally and drives it with
    // `isOpen`, so the component never unmounts between openings.
    const setOpen = (isOpen: boolean) =>
      view.rerender(
        <QueryClientProvider client={queryClient}>
          <AddModelsModal
            isOpen={isOpen}
            connection={connection}
            existingNames={existingNames}
            onClose={onClose}
          />
        </QueryClientProvider>,
      );
    return { ...view, setOpen };
  };

  const showUnverified = async () => {
    await userEvent.click(screen.getByTestId("add-models-verified-only"));
    await screen.findByTestId("add-models-row-openai/unverified-model");
  };

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    // `searchModels` is provider-scoped; the default fixture covers openai.
    vi.mocked(ConfigService.searchModels).mockResolvedValue(OPENAI_MODELS);
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue({
      name: "x",
      message: "ok",
    });
  });

  it("renders nothing without a connection to bind to", () => {
    renderModal(null);
    expect(screen.queryByTestId("add-models-modal")).not.toBeInTheDocument();
  });

  it("shows the connection summary and no provider combobox", async () => {
    renderModal();
    expect(
      await screen.findByTestId("add-models-connection-summary"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("add-models-provider")).not.toBeInTheDocument();
    expect(screen.getByText("Shared OpenAI")).toBeInTheDocument();
  });

  it("lists verified models for the connection's provider with derived names", async () => {
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    expect(
      screen.getByTestId("add-models-row-openai/gpt-4o-mini"),
    ).toBeInTheDocument();
    // unverified filtered out by default
    expect(
      screen.queryByTestId("add-models-row-openai/unverified-model"),
    ).not.toBeInTheDocument();
    // derived name pre-fills the input
    expect(
      screen.getByTestId("add-models-name-openai/gpt-4o-mini"),
    ).toHaveValue("gpt-4o-mini");
  });

  it("shows unverified models when the filter is toggled off", async () => {
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await userEvent.click(screen.getByTestId("add-models-verified-only"));
    await screen.findByTestId("add-models-row-openai/unverified-model");
  });

  it("distinguishes a provider with no models from a filter hiding them all", async () => {
    vi.mocked(ConfigService.searchModels).mockResolvedValue({
      items: [model("openai", "unverified-only", false)],
      next_page_id: null,
    });
    renderModal();
    await screen.findByTestId("add-models-empty");
    expect(screen.getByTestId("add-models-empty")).toHaveTextContent(
      "No results found",
    );
    await userEvent.click(screen.getByTestId("add-models-verified-only"));
    await screen.findByTestId("add-models-row-openai/unverified-only");
  });

  it("says the provider is empty when it really has no models", async () => {
    vi.mocked(ConfigService.searchModels).mockResolvedValue({
      items: [],
      next_page_id: null,
    });
    renderModal();
    await screen.findByTestId("add-models-empty");
    expect(screen.getByTestId("add-models-empty")).toHaveTextContent(
      "No models found for this provider.",
    );
  });

  it("flags names that collide with existing profiles and excludes them", async () => {
    renderModal(OPENAI_CONNECTION, ["gpt-4o-mini"]);
    await screen.findByTestId("add-models-row-openai/gpt-4o-mini");
    expect(
      screen.getByTestId("add-models-conflict-openai/gpt-4o-mini"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("add-models-check-openai/gpt-4o-mini"),
    ).toBeDisabled();
    // select-all reaches only the non-colliding row
    await userEvent.click(screen.getByTestId("add-models-select-all"));
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 1 profiles",
    );
  });

  it("selects nothing until the user chooses", async () => {
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    expect(
      screen.getByTestId("add-models-check-openai/gpt-4o"),
    ).not.toBeChecked();
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 0 profiles",
    );
    expect(screen.getByTestId("add-models-submit")).toBeDisabled();
  });

  it("creates profiles linked to the connection", async () => {
    const onClose = vi.fn();
    renderModal(OPENAI_CONNECTION, [], onClose);
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await userEvent.click(screen.getByTestId("add-models-check-openai/gpt-4o"));
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(ProfilesService.saveProfile).toHaveBeenCalledTimes(1);
    expect(ProfilesService.saveProfile).toHaveBeenCalledWith("gpt-4o", {
      llm: {
        model: "openai/gpt-4o",
        provider_connection_id: "conn-openai",
      },
      include_secrets: false,
    });
  });

  it("keeps the modal open and marks the row on per-model failure", async () => {
    const onClose = vi.fn();
    vi.mocked(ProfilesService.saveProfile)
      .mockResolvedValueOnce({ name: "a", message: "ok" })
      .mockRejectedValueOnce(new Error("boom"));
    renderModal(OPENAI_CONNECTION, [], onClose);
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );
    expect(onClose).not.toHaveBeenCalled();
    const rows = screen.getAllByText(/^(Saved|Failed)$/);
    expect(rows).toHaveLength(2);
  });

  it("stops the run and reports the server's reason when it refuses with 409", async () => {
    const onClose = vi.fn();
    vi.mocked(ProfilesService.saveProfile).mockRejectedValue(
      httpError(409, "Profile limit reached (10). Delete a profile first."),
    );
    renderModal(OPENAI_CONNECTION, [], onClose);
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await showUnverified();

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );
    // A second refusal confirms the wall; the third row is never attempted.
    expect(ProfilesService.saveProfile).toHaveBeenCalledTimes(2);
    expect(displayErrorToast).toHaveBeenCalledWith(
      "Profile limit reached (10). Delete a profile first.",
    );
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getAllByText(/^(Saved|Failed)$/)).toHaveLength(2);
    expect(screen.queryByTestId("loading-spinner")).not.toBeInTheDocument();
  });

  it("carries on past a single 409 so one raced name collision cannot halt the run", async () => {
    vi.mocked(ProfilesService.saveProfile)
      .mockRejectedValueOnce(httpError(409, "Profile 'x' already exists."))
      .mockResolvedValueOnce({ name: "b", message: "ok" });
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );
    expect(ProfilesService.saveProfile).toHaveBeenCalledTimes(2);
    expect(screen.getByText("Saved")).toBeInTheDocument();
    expect(displayErrorToast).toHaveBeenCalledWith(
      "Added 1 profiles; 1 failed",
    );
  });

  it("reports the blocking 409's own reason, not the earlier one's", async () => {
    vi.mocked(ProfilesService.saveProfile)
      .mockRejectedValueOnce(httpError(409, "Profile 'x' already exists."))
      .mockRejectedValueOnce(httpError(409));
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await showUnverified();

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );
    expect(displayErrorToast).toHaveBeenCalledWith(
      "Added 0 profiles; the server refused the rest",
    );
    expect(displayErrorToast).not.toHaveBeenCalledWith(
      "Profile 'x' already exists.",
    );
  });

  it("names the refusal generically when the 409 carries no detail", async () => {
    vi.mocked(ProfilesService.saveProfile).mockRejectedValue(httpError(409));
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await showUnverified();

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );
    expect(displayErrorToast).toHaveBeenCalledWith(
      "Added 0 profiles; the server refused the rest",
    );
  });

  it("keeps selections and edited names across a filter toggle", async () => {
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    const name = screen.getByTestId("add-models-name-openai/gpt-4o-mini");
    await userEvent.clear(name);
    await userEvent.type(name, "my-mini");
    await userEvent.click(
      screen.getByTestId("add-models-check-openai/gpt-4o-mini"),
    );

    await showUnverified();

    expect(
      screen.getByTestId("add-models-name-openai/gpt-4o-mini"),
    ).toHaveValue("my-mini");
    expect(
      screen.getByTestId("add-models-check-openai/gpt-4o-mini"),
    ).toBeChecked();

    await userEvent.click(screen.getByTestId("add-models-verified-only"));
    await waitFor(() =>
      expect(
        screen.queryByTestId("add-models-row-openai/unverified-model"),
      ).not.toBeInTheDocument(),
    );
    expect(
      screen.getByTestId("add-models-name-openai/gpt-4o-mini"),
    ).toHaveValue("my-mini");
  });

  it("starts a fresh session when the modal is reopened", async () => {
    vi.mocked(ProfilesService.saveProfile).mockRejectedValue(new Error("boom"));
    const { setOpen } = renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));
    await waitFor(() => expect(screen.getAllByText("Failed")).toHaveLength(2));

    setOpen(false);
    setOpen(true);

    expect(screen.queryByText("Failed")).not.toBeInTheDocument();
  });

  it("select-all toggles every selectable row", async () => {
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 0 profiles",
    );
    expect(screen.getByTestId("add-models-submit")).toBeDisabled();

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 2 profiles",
    );

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 0 profiles",
    );
  });
});
