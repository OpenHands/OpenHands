import { screen, fireEvent, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import ProfilesService, {
  type ProfileInfo,
} from "#/api/profiles-service/profiles-service.api";
import ProviderConnectionsService, {
  type ProviderConnection,
} from "#/api/provider-connections-service/provider-connections-service.api";
import { I18nKey } from "#/i18n/declaration";
import { LlmProfilesManager } from "./llm-profiles-manager";

const displayErrorToast = vi.hoisted(() => vi.fn());
const displaySuccessToast = vi.hoisted(() => vi.fn());

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast,
  displaySuccessToast,
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "backend-1", kind: "local" },
    orgId: null,
  }),
}));

vi.mock("#/hooks/query/use-search-providers", () => ({
  useSearchProviders: () => ({
    data: [
      { name: "openai", verified: true },
      { name: "anthropic", verified: true },
      { name: "openhands", verified: true },
      { name: "azure", verified: false },
    ],
  }),
}));

// The add-models modal hydrates its list through useProviderModels, which
// pages ConfigService.searchModels; stub it to an empty list so the modal
// renders its empty state without hitting a backend. The returned array is
// module-level so its reference is stable across renders — the modal's row
// effect keys off `data`, so a fresh array each render would loop forever.
const STUB_MODELS: unknown[] = [];
vi.mock("#/hooks/query/use-provider-models", () => ({
  useProviderModels: () => ({ data: STUB_MODELS, isLoading: false }),
}));

const connection: ProviderConnection = {
  id: "conn-1",
  display_name: "My OpenAI",
  provider: "openai",
  base_url: null,
  created_at: 1,
  updated_at: 2,
  api_key_set: true,
};

const emptyConnection: ProviderConnection = {
  ...connection,
  id: "conn-2",
  display_name: "My Anthropic",
  provider: "anthropic",
};

function makeProfile(overrides: Partial<ProfileInfo> = {}): ProfileInfo {
  return {
    name: "gpt-linked",
    model: "openai/gpt-4o",
    base_url: null,
    api_key_set: true,
    provider_connection_id: "conn-1",
    ...overrides,
  };
}

const renderManager = ({
  connections = [connection],
  profiles = [makeProfile()],
}: {
  connections?: ProviderConnection[];
  profiles?: ProfileInfo[];
} = {}) => {
  vi.spyOn(ProviderConnectionsService, "list").mockResolvedValue(connections);
  vi.spyOn(ProfilesService, "listProfiles").mockResolvedValue({
    profiles,
    active_profile: null,
  });
  return renderWithProviders(<LlmProfilesManager onEditProfile={vi.fn()} />);
};

const findGroup = async (label: string) => {
  const groups = await screen.findAllByTestId("provider-connection-group");
  const group = groups.find(
    (g) => within(g).getByTestId("profile-group-header").textContent === label,
  );
  if (!group) throw new Error(`No group labelled ${label}`);
  return group;
};

describe("Provider connection groups", () => {
  beforeEach(() => {
    displayErrorToast.mockReset();
    displaySuccessToast.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("nests each connection's profiles under it and lists unlinked profiles last", async () => {
    renderManager({
      connections: [connection, emptyConnection],
      profiles: [
        makeProfile(),
        makeProfile({ name: "standalone", provider_connection_id: null }),
        makeProfile({
          name: "orphaned",
          provider_connection_id: "conn-deleted",
          provider_connection_broken: true,
        }),
      ],
    });

    const linked = await findGroup("My OpenAI");
    expect(within(linked).getByText("openai")).toBeInTheDocument();
    await userEvent.click(
      within(linked).getByTestId("provider-connection-group-toggle"),
    );
    expect(within(linked).getByText("gpt-linked")).toBeInTheDocument();

    const empty = await findGroup("My Anthropic");
    await userEvent.click(
      within(empty).getByTestId("provider-connection-group-toggle"),
    );
    expect(
      within(empty).getByTestId("provider-connection-group-empty"),
    ).toBeInTheDocument();

    const headers = screen.getAllByTestId("profile-group-header");
    expect(headers.map((h) => h.textContent)).toEqual([
      "My OpenAI",
      "My Anthropic",
      I18nKey.SETTINGS$PROFILES_UNGROUPED,
    ]);
    // A profile whose connection was deleted joins the unlinked group, where
    // its broken-connection badge explains why.
    const unlinked = await findGroup(I18nKey.SETTINGS$PROFILES_UNGROUPED);
    await userEvent.click(
      within(unlinked).getByTestId("provider-connection-group-toggle"),
    );
    expect(within(unlinked).getByText("standalone")).toBeInTheDocument();
    expect(within(unlinked).getByText("orphaned")).toBeInTheDocument();
    expect(
      within(unlinked).getByTestId("profile-broken-connection-badge"),
    ).toBeInTheDocument();
    expect(
      within(unlinked).queryByTestId("provider-connection-menu-trigger"),
    ).not.toBeInTheDocument();
    expect(
      within(unlinked).queryByTestId("provider-connection-group-add-models"),
    ).not.toBeInTheDocument();
  });

  it("starts connection groups collapsed and toggles their profiles", async () => {
    const user = userEvent.setup();
    renderManager();

    const group = await findGroup("My OpenAI");
    const toggle = within(group).getByTestId(
      "provider-connection-group-toggle",
    );
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(within(group).queryByText("gpt-linked")).not.toBeInTheDocument();

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(within(group).getByText("gpt-linked")).toBeInTheDocument();

    await user.click(toggle);
    expect(within(group).queryByText("gpt-linked")).not.toBeInTheDocument();
  });

  it("opens Add models preselected to the connection from its menu", async () => {
    const user = userEvent.setup();
    renderManager();

    const group = await findGroup("My OpenAI");
    await user.click(
      within(group).getByTestId("provider-connection-menu-trigger"),
    );
    await user.click(screen.getByTestId("provider-connection-add-models"));

    expect(await screen.findByTestId("add-models-modal")).toBeInTheDocument();
    expect(screen.getByTestId("add-models-connection")).toHaveValue("conn-1");
  });

  it("offers Add models from a connection's header without expanding it", async () => {
    const user = userEvent.setup();
    renderManager({ connections: [emptyConnection], profiles: [] });

    const group = await findGroup("My Anthropic");
    await user.click(
      within(group).getByTestId("provider-connection-group-add-models"),
    );

    expect(await screen.findByTestId("add-models-modal")).toBeInTheDocument();
    expect(screen.getByTestId("add-models-connection")).toHaveValue("conn-2");
    expect(
      within(group).getByTestId("provider-connection-group-toggle"),
    ).toHaveAttribute("aria-expanded", "false");
  });

  it("shows supported providers in the create-connection selector", async () => {
    const user = userEvent.setup();
    renderManager({ connections: [], profiles: [] });

    await user.click(await screen.findByTestId("add-provider-connection"));

    const providerSelector = screen.getByRole("combobox", {
      name: /provider/i,
    });
    await user.click(providerSelector);

    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByText("Anthropic")).toBeInTheDocument();
    expect(screen.getByText("OpenHands")).toBeInTheDocument();
    expect(screen.getByText("Azure")).toBeInTheDocument();

    await user.click(screen.getByText("Anthropic"));
    expect(providerSelector).toHaveValue("Anthropic");
  });

  it("submits the raw provider id when creating a connection", async () => {
    const user = userEvent.setup();
    const createSpy = vi
      .spyOn(ProviderConnectionsService, "create")
      .mockResolvedValue(emptyConnection);
    renderManager({ connections: [], profiles: [] });

    await user.click(await screen.findByTestId("add-provider-connection"));
    await user.type(
      screen.getByTestId("provider-connection-name-input"),
      "My Anthropic",
    );

    const providerSelector = screen.getByRole("combobox", {
      name: /provider/i,
    });
    await user.click(providerSelector);
    await user.click(screen.getByTestId("provider-item-anthropic"));

    await user.type(
      screen.getByTestId("provider-connection-api-key-input"),
      "test-key",
    );
    await user.click(screen.getByTestId("provider-connection-submit"));

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith(
        expect.objectContaining({ provider: "anthropic" }),
      );
    });
  });

  it("shows supported providers in the edit-connection selector", async () => {
    const user = userEvent.setup();
    renderManager();

    await user.click(
      await screen.findByTestId("provider-connection-menu-trigger"),
    );
    await user.click(screen.getByTestId("provider-connection-edit"));

    const providerSelector = screen.getByRole("combobox", {
      name: /provider/i,
    });
    expect(providerSelector).toHaveValue("OpenAI");

    await user.click(providerSelector);

    expect(screen.getByText("Anthropic")).toBeInTheDocument();
    expect(screen.getByText("OpenHands")).toBeInTheDocument();
  });

  it("surfaces the server message when deleting a referenced connection fails", async () => {
    const conflict = Object.assign(new Error("HTTP 409"), {
      response: {
        detail: "Connection is used by profile 'gpt-4o'.",
      },
    });
    const deleteSpy = vi
      .spyOn(ProviderConnectionsService, "delete")
      .mockRejectedValue(conflict);
    renderManager();

    fireEvent.click(
      await screen.findByTestId("provider-connection-menu-trigger"),
    );
    fireEvent.click(screen.getByTestId("provider-connection-delete"));
    fireEvent.click(screen.getByTestId("delete-provider-connection-confirm"));

    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalledWith(
        "Connection is used by profile 'gpt-4o'.",
      );
    });
    expect(deleteSpy).toHaveBeenCalledWith("conn-1");
  });
});
