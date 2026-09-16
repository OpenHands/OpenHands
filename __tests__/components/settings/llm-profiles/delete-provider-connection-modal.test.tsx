import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DeleteProviderConnectionModal } from "#/components/features/settings/llm-profiles/delete-provider-connection-modal";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import ProviderConnectionsService from "#/api/provider-connections-service/provider-connections-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string | number>) => {
      const translations: Record<string, string> = {
        SETTINGS$PROVIDER_CONNECTION_DELETE_TITLE: "Delete provider connection",
        SETTINGS$PROVIDER_CONNECTION_DELETE_CONFIRMATION: "Delete {{name}}?",
        SETTINGS$PROVIDER_CONNECTION_DELETE_BLOCKED:
          "Can't delete — {{count}} profile(s) still use this connection. Unlink or delete them first.",
        BUTTON$CANCEL: "Cancel",
        BUTTON$DELETE: "Delete",
      };
      let result = translations[key] || key;
      if (params) {
        for (const [paramKey, value] of Object.entries(params)) {
          result = result.replace(`{{${paramKey}}}`, String(value));
        }
      }
      return result;
    },
  }),
}));

vi.mock("#/api/provider-connections-service/provider-connections-service.api");

const connection: ProviderConnection = {
  id: "conn-1",
  display_name: "Ollama Cloud",
  provider: "openai",
  base_url: "https://ollama.com/v1",
  created_at: 1,
  updated_at: 2,
  api_key_set: true,
};

describe("DeleteProviderConnectionModal", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  const renderModal = (linkedProfileCount = 0) =>
    render(
      <QueryClientProvider client={queryClient}>
        <DeleteProviderConnectionModal
          connection={connection}
          linkedProfileCount={linkedProfileCount}
          onClose={vi.fn()}
        />
      </QueryClientProvider>,
    );

  it("allows deletion and shows no warning when no profiles are linked", () => {
    renderModal(0);
    expect(
      screen.queryByTestId("delete-provider-connection-blocked"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("delete-provider-connection-confirm"),
    ).not.toBeDisabled();
  });

  it("blocks deletion and shows the linked profile count as a warning", () => {
    renderModal(3);
    expect(
      screen.getByTestId("delete-provider-connection-blocked"),
    ).toHaveTextContent(
      "Can't delete — 3 profile(s) still use this connection. Unlink or delete them first.",
    );
    expect(
      screen.getByTestId("delete-provider-connection-confirm"),
    ).toBeDisabled();
  });

  it("never calls delete when blocked, even if the disabled button is clicked", async () => {
    const user = userEvent.setup();
    renderModal(2);

    await user.click(screen.getByTestId("delete-provider-connection-confirm"));

    expect(ProviderConnectionsService.delete).not.toHaveBeenCalled();
  });
});
