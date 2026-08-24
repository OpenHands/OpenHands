import { screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import ProviderConnectionsService, {
  type ProviderConnection,
} from "#/api/provider-connections-service/provider-connections-service.api";
import { ProviderConnectionsManager } from "./provider-connections-manager";

const displayErrorToast = vi.hoisted(() => vi.fn());
const displaySuccessToast = vi.hoisted(() => vi.fn());

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast,
  displaySuccessToast,
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

const renderWith = (ui: React.ReactElement) => renderWithProviders(ui);

const connection: ProviderConnection = {
  id: "conn-1",
  display_name: "My OpenAI",
  provider: "openai",
  base_url: null,
  created_at: 1,
  updated_at: 2,
  api_key_set: true,
};

describe("ProviderConnectionsManager", () => {
  beforeEach(() => {
    displayErrorToast.mockReset();
    displaySuccessToast.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows an empty state when there are no connections", () => {
    renderWith(
      <ProviderConnectionsManager
        connections={[]}
        linkedCountById={{}}
        isLoading={false}
        loadError={null}
      />,
    );

    expect(
      screen.getByTestId("provider-connections-empty"),
    ).toBeInTheDocument();
  });

  it("shows supported providers in the create-connection selector", async () => {
    const user = userEvent.setup();

    renderWith(
      <ProviderConnectionsManager
        connections={[]}
        linkedCountById={{}}
        isLoading={false}
        loadError={null}
      />,
    );

    await user.click(screen.getByTestId("add-provider-connection"));

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
      .mockResolvedValue({
        ...connection,
        id: "conn-anthropic",
        display_name: "My Anthropic",
        provider: "anthropic",
      });

    renderWith(
      <ProviderConnectionsManager
        connections={[]}
        linkedCountById={{}}
        isLoading={false}
        loadError={null}
      />,
    );

    await user.click(screen.getByTestId("add-provider-connection"));
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

  it("lists a row per connection with its display name and provider", () => {
    renderWith(
      <ProviderConnectionsManager
        connections={[connection]}
        linkedCountById={{ "conn-1": 3 }}
        isLoading={false}
        loadError={null}
      />,
    );

    expect(screen.getByTestId("provider-connection-row")).toBeInTheDocument();
    expect(screen.getByText("My OpenAI")).toBeInTheDocument();
    expect(screen.getByText("openai")).toBeInTheDocument();
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

    renderWith(
      <ProviderConnectionsManager
        connections={[connection]}
        linkedCountById={{ "conn-1": 1 }}
        isLoading={false}
        loadError={null}
      />,
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
