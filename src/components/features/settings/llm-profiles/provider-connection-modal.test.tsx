import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import {
  isValidBaseUrl,
  ProviderConnectionModal,
} from "./provider-connection-modal";

vi.mock("#/hooks/query/use-search-providers", () => ({
  useSearchProviders: () => ({
    data: [
      { name: "openai", verified: true },
      { name: "anthropic", verified: true },
    ],
  }),
}));

const mutateAsync = vi.fn();

vi.mock("#/hooks/mutation/use-create-provider-connection", () => ({
  useCreateProviderConnection: () => ({ mutateAsync, isPending: false }),
}));

vi.mock("#/hooks/mutation/use-update-provider-connection", () => ({
  useUpdateProviderConnection: () => ({ mutateAsync, isPending: false }),
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

const renderModal = () =>
  renderWithProviders(
    <ProviderConnectionModal
      connection={connection}
      isCreate={false}
      onClose={vi.fn()}
    />,
  );

const submitButton = () =>
  screen.getByTestId("provider-connection-submit") as HTMLButtonElement;

describe("isValidBaseUrl", () => {
  it.each([
    ["", true], // optional: empty is valid
    ["https://api.openai.com", true],
    ["http://localhost:8080/v1", true],
    ["api.openai.com", false], // missing protocol
    ["ftp://example.com", false], // wrong protocol
    ["not a url", false],
    ["https://", false], // empty hostname
  ])("isValidBaseUrl(%j) === %s", (value, expected) => {
    expect(isValidBaseUrl(value)).toBe(expected);
  });
});

describe("ProviderConnectionModal base URL validation", () => {
  it("allows saving when the optional base URL is empty", () => {
    renderModal();
    expect(
      screen.queryByTestId("provider-connection-base-url-input-error"),
    ).not.toBeInTheDocument();
    expect(submitButton().disabled).toBe(false);
  });

  it("blocks saving and shows an error for a malformed base URL", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(
      screen.getByTestId("provider-connection-base-url-input"),
      "not-a-url",
    );

    const error = screen.getByTestId(
      "provider-connection-base-url-input-error",
    );
    expect(error).toBeInTheDocument();
    expect(error).toHaveTextContent(
      "SETTINGS$PROVIDER_CONNECTION_INVALID_BASE_URL",
    );
    expect(submitButton().disabled).toBe(true);
  });

  it("recovers when the base URL is corrected", async () => {
    const user = userEvent.setup();
    renderModal();

    const input = screen.getByTestId("provider-connection-base-url-input");
    await user.type(input, "not-a-url");
    expect(submitButton().disabled).toBe(true);

    await user.clear(input);
    await user.type(input, "https://api.openai.com/v1");
    expect(
      screen.queryByTestId("provider-connection-base-url-input-error"),
    ).not.toBeInTheDocument();
    expect(submitButton().disabled).toBe(false);
  });
});
