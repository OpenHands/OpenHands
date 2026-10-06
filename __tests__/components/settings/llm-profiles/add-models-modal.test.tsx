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
  LLMProvider,
  ProviderPage,
} from "#/api/config-service/config-service.types";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) => {
      const translations: Record<string, string> = {
        SETTINGS$ADD_MODELS_TITLE: "Add models as profiles",
        SETTINGS$ADD_MODELS_PROVIDER_LABEL: "Provider",
        SETTINGS$ADD_MODELS_PROVIDER_PLACEHOLDER: "Select a provider",
        SETTINGS$ADD_MODELS_CONNECTION_LABEL: "Provider connection",
        SETTINGS$ADD_MODELS_NO_CONNECTION: "No connection (keyless)",
        SETTINGS$ADD_MODELS_CONNECTION_BOUND: `Linked to ${params?.provider ?? "?"}`,
        SETTINGS$ADD_MODELS_KEYLESS_NOTE:
          "No connection — profiles will need a key added later",
        SETTINGS$ADD_MODELS_SELECT_ALL: "Select all",
        SETTINGS$ADD_MODELS_EMPTY: "No models found for this provider.",
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

// The provider combobox is driven by `searchProviders`. The modal only needs
// the openai provider to appear so it can be picked / preselected.
const PROVIDERS: ProviderPage = {
  items: [{ name: "openai", verified: true } satisfies LLMProvider],
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
    // When a connection is given, preselect it (the "..." entry point). When
    // null, render chooser mode with no preselect (the top-button entry
    // point) and an empty connection pool — the modal then just shows the
    // combobox prompt.
    const connections = connection ? [connection] : [];
    const initialConnectionId = connection?.id ?? null;
    const view = render(
      <QueryClientProvider client={queryClient}>
        <AddModelsModal
          isOpen
          connections={connections}
          initialConnectionId={initialConnectionId}
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
            connections={connections}
            initialConnectionId={initialConnectionId}
            existingNames={existingNames}
            onClose={onClose}
          />
        </QueryClientProvider>,
      );
    return { ...view, setOpen };
  };

  const showUnverified = async () => {
    // No verified filter anymore — unverified models are always visible, so
    // this just waits for the unverified row to render. Kept as a helper so
    // call sites read clearly.
    await screen.findByTestId("add-models-row-openai/unverified-model");
  };

  // The provider combobox options arrive asynchronously (searchProviders is a
  // query), so wait for the openai option before interacting with it.
  const chooseOpenAI = async () => {
    await screen.findByText("OpenAI");
    await userEvent.selectOptions(
      screen.getByTestId("add-models-provider"),
      "openai",
    );
  };

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    // `searchModels` is provider-scoped; the default fixture covers openai.
    vi.mocked(ConfigService.searchModels).mockResolvedValue(OPENAI_MODELS);
    // `searchProviders` feeds the provider combobox.
    vi.mocked(ConfigService.searchProviders).mockResolvedValue(PROVIDERS);
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue({
      name: "x",
      message: "ok",
    });
  });

  it("renders only the provider combobox in chooser mode (no provider picked)", () => {
    renderModal(null);
    expect(screen.getByTestId("add-models-modal")).toBeInTheDocument();
    // The provider box is the prompt — no model list, no empty-state, no
    // spinner, and no connection box (there is no provider to match yet).
    expect(screen.getByTestId("add-models-provider")).toBeInTheDocument();
    expect(screen.queryByTestId("add-models-loading")).not.toBeInTheDocument();
    expect(screen.queryByTestId("add-models-empty")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("add-models-connection-field"),
    ).not.toBeInTheDocument();
  });

  it("shows the provider preselected to the launched connection's provider", async () => {
    renderModal();
    // Preselect mode: the provider box opens on the connection's provider,
    // the connection is bound, and the models load without the user choosing.
    await screen.findByText("OpenAI");
    const providerBox = screen.getByTestId(
      "add-models-provider",
    ) as HTMLSelectElement;
    expect(providerBox.value).toBe("openai");
    const connectionBox = (await screen.findByTestId(
      "add-models-connection",
    )) as HTMLSelectElement;
    expect(connectionBox.value).toBe("conn-openai");
    await screen.findByTestId("add-models-row-openai/gpt-4o");
  });

  it("lists all models for the connection's provider with short (unprefixed) names", async () => {
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    expect(
      screen.getByTestId("add-models-row-openai/gpt-4o-mini"),
    ).toBeInTheDocument();
    // No verified filter: unverified models are shown too.
    expect(
      screen.getByTestId("add-models-row-openai/unverified-model"),
    ).toBeInTheDocument();
    // The provider prefix is stripped from the displayed label — it's implied
    // by the selected connection.
    expect(screen.getByText("gpt-4o-mini")).toBeInTheDocument();
    expect(screen.queryByText("openai/gpt-4o-mini")).not.toBeInTheDocument();
    // No per-row name editor.
    expect(
      screen.queryByTestId("add-models-name-openai/gpt-4o-mini"),
    ).not.toBeInTheDocument();
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

  it("hides models whose name matches an existing profile", async () => {
    renderModal(OPENAI_CONNECTION, ["gpt-4o-mini"]);
    // The non-hidden model renders first; await it so the query has settled.
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    // The already-added model is not rendered at all — no row, no checkbox,
    // no conflict badge. Showing it disabled just clutters the list.
    expect(
      screen.queryByTestId("add-models-row-openai/gpt-4o-mini"),
    ).not.toBeInTheDocument();
    // Preselect mode auto-selects the visible rows (gpt-4o + unverified-model);
    // the hidden gpt-4o-mini is not counted.
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 2 profiles",
    );
  });

  it("dedupes a model the catalog lists twice (verified + raw)", async () => {
    // A provider's catalog can surface the same model twice — once in the
    // verified set and once in the raw list — when the two disagree on
    // prefixing. Both must collapse to a single row instead of flagging each
    // other as a name conflict on a fresh account with no profiles.
    vi.mocked(ConfigService.searchModels).mockResolvedValue({
      items: [
        model("openai", "gpt-4o", true),
        model("openai", "gpt-4o", false),
        model("openai", "gpt-4o-mini", true),
      ],
      next_page_id: null,
    });
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    expect(screen.getAllByTestId("add-models-row-openai/gpt-4o")).toHaveLength(
      1,
    );
    expect(
      screen.queryByTestId("add-models-conflict-openai/gpt-4o"),
    ).not.toBeInTheDocument();
  });

  it("selects nothing until the user chooses (chooser mode)", async () => {
    // Chooser entry point: no preselect, so the user picks a provider. Rows
    // are not auto-selected — choosing is the point of the modal.
    renderModal(null);
    await chooseOpenAI();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    expect(
      screen.getByTestId("add-models-check-openai/gpt-4o"),
    ).not.toBeChecked();
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 0 profiles",
    );
    expect(screen.getByTestId("add-models-submit")).toBeDisabled();
  });

  it("auto-selects the loaded rows in preselect mode (connection row entry point)", async () => {
    // The "..." entry point is an explicit intent to bulk-add from that
    // connection, so its models load already selected.
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    expect(screen.getByTestId("add-models-check-openai/gpt-4o")).toBeChecked();
    expect(
      screen.getByTestId("add-models-check-openai/gpt-4o-mini"),
    ).toBeChecked();
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 3 profiles",
    );
  });

  it("creates profiles linked to the connection", async () => {
    const onClose = vi.fn();
    renderModal(OPENAI_CONNECTION, [], onClose);
    // Preselect mode auto-selects the rows; submit creates one profile per
    // selected row, each linked to the bound connection.
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(ProfilesService.saveProfile).toHaveBeenCalledTimes(3);
    expect(ProfilesService.saveProfile).toHaveBeenCalledWith("gpt-4o", {
      llm: {
        model: "openai/gpt-4o",
        provider_connection_id: "conn-openai",
      },
      include_secrets: false,
    });
  });

  it("creates keyless profiles when the provider has no matching connection", async () => {
    // Chooser entry point with no connections at all: the user picks a
    // provider from the catalog and bulk-adds. With no connection to bind,
    // profiles are created keyless (provider_connection_id: null) — the
    // pre-connection bulk-add behavior the linked issue requires.
    const onClose = vi.fn();
    renderModal(null, [], onClose);
    await chooseOpenAI();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    // No connection box: nothing matches the chosen provider.
    expect(
      screen.queryByTestId("add-models-connection-field"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("add-models-keyless-note")).toHaveTextContent(
      "No connection — profiles will need a key added later",
    );
    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(ProfilesService.saveProfile).toHaveBeenCalledWith("gpt-4o", {
      llm: {
        model: "openai/gpt-4o",
        provider_connection_id: null,
      },
      include_secrets: false,
    });
  });

  it("binds a matching connection by default when the user picks a provider in chooser mode", async () => {
    // Picking a provider that has a connection binds it automatically, so
    // created profiles are usable out of the box. The user can opt out via
    // the connection box's "No connection" option.
    const onClose = vi.fn();
    // Chooser entry point (no preselect) but a connection exists in the pool.
    render(
      <QueryClientProvider client={queryClient}>
        <AddModelsModal
          isOpen
          connections={[OPENAI_CONNECTION]}
          initialConnectionId={null}
          existingNames={[]}
          onClose={onClose}
        />
      </QueryClientProvider>,
    );
    await chooseOpenAI();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    // The connection box appears and defaults to the matching connection.
    const connectionBox = screen.getByTestId(
      "add-models-connection",
    ) as HTMLSelectElement;
    expect(connectionBox.value).toBe("conn-openai");
    expect(
      screen.queryByTestId("add-models-keyless-note"),
    ).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(ProfilesService.saveProfile).toHaveBeenCalledWith("gpt-4o", {
      llm: {
        model: "openai/gpt-4o",
        provider_connection_id: "conn-openai",
      },
      include_secrets: false,
    });
  });

  it("lets the user opt out of a matching connection to create keyless profiles", async () => {
    const onClose = vi.fn();
    render(
      <QueryClientProvider client={queryClient}>
        <AddModelsModal
          isOpen
          connections={[OPENAI_CONNECTION]}
          initialConnectionId={null}
          existingNames={[]}
          onClose={onClose}
        />
      </QueryClientProvider>,
    );
    await chooseOpenAI();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    // Explicitly choose "No connection (keyless)".
    await userEvent.selectOptions(
      screen.getByTestId("add-models-connection"),
      "",
    );
    expect(screen.getByTestId("add-models-keyless-note")).toBeInTheDocument();

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    await userEvent.click(screen.getByTestId("add-models-submit"));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(ProfilesService.saveProfile).toHaveBeenCalledWith("gpt-4o", {
      llm: {
        model: "openai/gpt-4o",
        provider_connection_id: null,
      },
      include_secrets: false,
    });
  });

  it("keeps the modal open and marks the row on per-model failure", async () => {
    const onClose = vi.fn();
    vi.mocked(ProfilesService.saveProfile)
      .mockResolvedValueOnce({ name: "a", message: "ok" })
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue({ name: "c", message: "ok" });
    renderModal(OPENAI_CONNECTION, [], onClose);
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    // Preselect mode auto-selects the rows; submit drives all three.
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );
    expect(onClose).not.toHaveBeenCalled();
    const rows = screen.getAllByText(/^(Saved|Failed)$/);
    expect(rows).toHaveLength(3);
  });

  it("stops the run and reports the server's reason when it refuses with 409", async () => {
    const onClose = vi.fn();
    vi.mocked(ProfilesService.saveProfile).mockRejectedValue(
      httpError(409, "Profile limit reached (10). Delete a profile first."),
    );
    renderModal(OPENAI_CONNECTION, [], onClose);
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await showUnverified();

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
      .mockResolvedValue({ name: "b", message: "ok" });
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );
    expect(ProfilesService.saveProfile).toHaveBeenCalledTimes(3);
    expect(screen.getAllByText("Saved")).toHaveLength(2);
    expect(displayErrorToast).toHaveBeenCalledWith(
      "Added 2 profiles; 1 failed",
    );
  });

  it("reports the blocking 409's own reason, not the earlier one's", async () => {
    vi.mocked(ProfilesService.saveProfile)
      .mockRejectedValueOnce(httpError(409, "Profile 'x' already exists."))
      .mockRejectedValueOnce(httpError(409));
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await showUnverified();

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

    await userEvent.click(screen.getByTestId("add-models-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );
    expect(displayErrorToast).toHaveBeenCalledWith(
      "Added 0 profiles; the server refused the rest",
    );
  });

  it("keeps selections once the list has settled", async () => {
    // Preselect mode auto-selects the rows; a selection must survive the
    // list settling (the unverified row arriving after the verified ones).
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await showUnverified();

    expect(
      screen.getByTestId("add-models-check-openai/gpt-4o-mini"),
    ).toBeChecked();
  });

  it("starts a fresh session when the modal is reopened", async () => {
    vi.mocked(ProfilesService.saveProfile).mockRejectedValue(new Error("boom"));
    const { setOpen } = renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    await userEvent.click(screen.getByTestId("add-models-submit"));
    await waitFor(() => expect(screen.getAllByText("Failed")).toHaveLength(3));

    setOpen(false);
    setOpen(true);

    expect(screen.queryByText("Failed")).not.toBeInTheDocument();
  });

  it("select-all toggles every selectable row", async () => {
    renderModal();
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    // Preselect mode auto-selects every row.
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 3 profiles",
    );

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 0 profiles",
    );
    expect(screen.getByTestId("add-models-submit")).toBeDisabled();

    await userEvent.click(screen.getByTestId("add-models-select-all"));
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 3 profiles",
    );
  });

  it("does not re-submit saved rows on retry (no false failure)", async () => {
    // Two models: the first saves, the second fails. The saved row must drop
    // out of the selection so a retry only re-attempts the failed one —
    // otherwise the saved row is re-created, 409s on its now-existing name,
    // and surfaces as a failure for work that already succeeded.
    vi.mocked(ConfigService.searchModels).mockResolvedValue({
      items: [
        model("openai", "gpt-4o", true),
        model("openai", "gpt-4o-mini", true),
      ],
      next_page_id: null,
    });
    vi.mocked(ProfilesService.saveProfile)
      .mockResolvedValueOnce({ name: "gpt-4o", message: "ok" })
      .mockRejectedValue(new Error("boom"));
    renderModal(OPENAI_CONNECTION, []);
    await screen.findByTestId("add-models-row-openai/gpt-4o");

    await userEvent.click(screen.getByTestId("add-models-submit"));
    await waitFor(() =>
      expect(screen.getByTestId("add-models-submit")).not.toBeDisabled(),
    );

    // The saved row is no longer selectable; only the failed row remains.
    expect(screen.getByTestId("add-models-submit")).toHaveTextContent(
      "Add 1 profiles",
    );

    // Retry: exactly one more create (the failed row), not two.
    await userEvent.click(screen.getByTestId("add-models-submit"));
    await waitFor(() =>
      expect(ProfilesService.saveProfile).toHaveBeenCalledTimes(3),
    );
    // The third create targets the failed row, never the saved one.
    expect(
      vi.mocked(ProfilesService.saveProfile).mock.calls[2][1].llm.model,
    ).toBe("openai/gpt-4o-mini");
  });

  it("reloads the model list when reopened in preselect mode", async () => {
    // Close leaves the provider set in state, and useProviderModels serves a
    // cached list for the same provider — so reopening used to render the
    // empty state with Submit disabled. Resetting provider/connection on close
    // makes the list reload.
    const { setOpen } = renderModal(OPENAI_CONNECTION, []);
    await screen.findByTestId("add-models-row-openai/gpt-4o");

    setOpen(false);
    setOpen(true);

    // The model list reloads instead of showing the empty state.
    await screen.findByTestId("add-models-row-openai/gpt-4o");
    expect(screen.queryByTestId("add-models-empty")).not.toBeInTheDocument();
    // And the rows are re-selected (preselect intent carries across the reopen).
    expect(screen.getByTestId("add-models-submit")).not.toBeDisabled();
  });
});
