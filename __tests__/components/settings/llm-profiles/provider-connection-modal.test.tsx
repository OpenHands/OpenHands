import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProviderConnectionModal } from "#/components/features/settings/llm-profiles/provider-connection-modal";

describe("ProviderConnectionModal", () => {
  it("shows the OpenRouter endpoint as an example without overwriting a saved gateway", async () => {
    render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ProviderConnectionModal
          isCreate={false}
          onClose={() => {}}
          connection={{
            id: "openrouter-connection",
            display_name: "OpenRouter",
            provider: "openrouter",
            base_url: "https://gateway.example/v1",
            api_key_set: true,
            created_at: 0,
            updated_at: 0,
          }}
        />
      </QueryClientProvider>,
    );
    const input = await screen.findByTestId(
      "provider-connection-base-url-input",
    );
    await waitFor(() =>
      expect(input).toHaveAttribute(
        "placeholder",
        "https://openrouter.ai/api/v1",
      ),
    );
    expect(input).toHaveValue("https://gateway.example/v1");
  });
});
