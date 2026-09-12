import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProviderConnectionRow } from "#/components/features/settings/llm-profiles/provider-connection-row";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string | number>) => {
      const translations: Record<string, string> = {
        SETTINGS$PROVIDER_CONNECTION_MODEL_COUNT: "{{count}} model(s)",
        SETTINGS$PROVIDER_CONNECTION_KEY_SET_LABEL:
          "API key stored (not verified against the provider)",
        SETTINGS$PROVIDER_CONNECTION_KEY_UNSET_LABEL: "No API key stored",
        SETTINGS$PROVIDER_CONNECTION_BULK_ADD_ACTION: "Bulk add models",
        SETTINGS$PROVIDER_CONNECTION_EDIT_TITLE: "Edit provider connection",
        SETTINGS$PROVIDER_CONNECTION_DELETE_TITLE: "Delete provider connection",
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

const baseConnection: ProviderConnection = {
  id: "conn-1",
  display_name: "Ollama Cloud",
  provider: "openai",
  base_url: "https://ollama.com/v1",
  created_at: 1,
  updated_at: 2,
  api_key_set: true,
};

describe("ProviderConnectionRow", () => {
  it("labels the key status icon as 'stored', not 'verified', when a key is set", () => {
    render(
      <ProviderConnectionRow
        connection={baseConnection}
        linkedProfileCount={3}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onBulkAddModels={vi.fn()}
      />,
    );

    expect(screen.getByTestId("set-indicator")).toHaveAttribute(
      "aria-label",
      "API key stored (not verified against the provider)",
    );
  });

  it("labels the key status icon as unset when no key is stored", () => {
    render(
      <ProviderConnectionRow
        connection={{ ...baseConnection, api_key_set: false }}
        linkedProfileCount={0}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onBulkAddModels={vi.fn()}
      />,
    );

    expect(screen.getByTestId("unset-indicator")).toHaveAttribute(
      "aria-label",
      "No API key stored",
    );
  });

  it("calls onBulkAddModels with the connection when the bulk-add button is clicked", () => {
    const onBulkAddModels = vi.fn();
    render(
      <ProviderConnectionRow
        connection={baseConnection}
        linkedProfileCount={0}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onBulkAddModels={onBulkAddModels}
      />,
    );

    screen.getByTestId("provider-connection-bulk-add").click();
    expect(onBulkAddModels).toHaveBeenCalledWith(baseConnection);
  });
});
