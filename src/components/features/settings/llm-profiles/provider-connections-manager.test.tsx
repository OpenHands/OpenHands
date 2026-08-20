import { screen, fireEvent, waitFor } from "@testing-library/react";
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
