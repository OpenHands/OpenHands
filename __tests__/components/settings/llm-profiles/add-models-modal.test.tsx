import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AddModelsModal } from "#/components/features/settings/llm-profiles/add-models-modal";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import ConfigService from "#/api/config-service/config-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

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

const PROVIDERS = {
  items: [
    { name: "openhands", verified: true },
    { name: "openai", verified: true },
    // The provider feed carries entries that are not providers at all; they
    // arrive unverified and belong under the "Others" divider.
    { name: "1024-x-1024", verified: false },
  ],
  next_page_id: null,
};

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

  const renderModal = (existingNames: string[] = [], onClose = vi.fn()) => {
    const view = render(
      <QueryClientProvider client={queryClient}>
        <AddModelsModal
          isOpen
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
            existingNames={existingNames}
            onClose={onClose}
          />
        </QueryClientProvider>,
      );
    return { ...view, setOpen };
  };

  const showUnverified = async () => {
    await userEvent.click(screen.getByTestId("add-models-verified-only"));
    await screen.findByTestId("add-models-row-openhands/unverified-model");
  };

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
    // select-all reaches only the non-colliding row
    await userEvent.click(screen.getByTestId("add-models-select-all"));
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 1 profiles",
    );
  });

  it("selects nothing until the user chooses", async () => {
    renderModal();
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );

    expect(
      screen.getByTestId("add-models-check-openhands/trinity-large-thinking"),
    ).not.toBeChecked();
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 0 profiles",
    );
    expect(screen.getByTestId("add-models-submit")).toBeDisabled();
  });

  it("groups unverified providers under a separate divider", async () => {
    renderModal();
    const select = await screen.findByTestId("add-models-provider");
    // options arrive with the providers query, not on first paint
    await screen.findByRole("option", { name: "1024-x-1024" });

    const groups = Array.from(select.querySelectorAll("optgroup"));
    expect(groups).toHaveLength(2);
    expect(
      Array.from(groups[0].querySelectorAll("option")).map((o) => o.value),
    ).toEqual(["openhands", "openai"]);
    expect(
      Array.from(groups[1].querySelectorAll("option")).map((o) => o.value),
    ).toEqual(["1024-x-1024"]);
  });

  it("creates one profile per selected model, keyless", async () => {
    const onClose = vi.fn();
    renderModal([], onClose);
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );

    // pick one; only that one is created
    await userEvent.click(
      screen.getByTestId("add-models-check-openhands/deepseek-v4-flash"),
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
    renderModal([], onClose);
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );
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
    // only the attempted rows report a status; the unattempted row is not
    // left spinning on a "saving" it never got
    expect(screen.getAllByText(/^(Saved|Failed)$/)).toHaveLength(2);
    expect(screen.queryByTestId("loading-spinner")).not.toBeInTheDocument();
  });

  it("carries on past a single 409 so one raced name collision cannot halt the run", async () => {
    // A name free at load can be taken by another session before submit. That
    // 409 is about one row, not the account's profile ceiling.
    vi.mocked(ProfilesService.saveProfile)
      .mockRejectedValueOnce(httpError(409, "Profile 'x' already exists."))
      .mockResolvedValueOnce({ name: "b", message: "ok" });
    renderModal();
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );

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

  it("names the refusal generically when the 409 carries no detail", async () => {
    // A body the client cannot quote must not fall through to a count that
    // omits the rows the server never let it attempt.
    vi.mocked(ProfilesService.saveProfile).mockRejectedValue(httpError(409));
    renderModal();
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );
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
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );

    const name = screen.getByTestId(
      "add-models-name-openhands/deepseek-v4-flash",
    );
    await userEvent.clear(name);
    await userEvent.type(name, "my-flash");
    await userEvent.click(
      screen.getByTestId("add-models-check-openhands/deepseek-v4-flash"),
    );

    await showUnverified();

    expect(
      screen.getByTestId("add-models-name-openhands/deepseek-v4-flash"),
    ).toHaveValue("my-flash");
    expect(
      screen.getByTestId("add-models-check-openhands/deepseek-v4-flash"),
    ).toBeChecked();

    // and back again
    await userEvent.click(screen.getByTestId("add-models-verified-only"));
    await waitFor(() =>
      expect(
        screen.queryByTestId("add-models-row-openhands/unverified-model"),
      ).not.toBeInTheDocument(),
    );
    expect(
      screen.getByTestId("add-models-name-openhands/deepseek-v4-flash"),
    ).toHaveValue("my-flash");
  });

  it("starts a fresh session when the modal is reopened", async () => {
    vi.mocked(ProfilesService.saveProfile).mockRejectedValue(new Error("boom"));
    const { setOpen } = renderModal();
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );
    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));
    await waitFor(() => expect(screen.getAllByText("Failed")).toHaveLength(2));

    setOpen(false);
    setOpen(true);

    const select = await screen.findByTestId("add-models-provider");
    expect(select).toHaveValue("");
    expect(screen.queryByText("Failed")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("add-models-row-openhands/deepseek-v4-flash"),
    ).not.toBeInTheDocument();
  });

  it("select-all toggles every selectable row", async () => {
    renderModal();
    await pickProvider();
    await screen.findByTestId(
      "add-models-row-openhands/trinity-large-thinking",
    );

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
