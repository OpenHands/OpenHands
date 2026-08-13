import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConnectProviderWizard } from "#/components/features/settings/provider-connections/connect-provider-wizard";
import type { ProviderConnection } from "#/api/provider-connections-service";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const mockProviders = [{ name: "openai", verified: true }];

// Catalog returned by useProviderModels: two recommended, one not.
const mockCatalog = [
  { provider: "openai", name: "gpt-4o", verified: true },
  { provider: "openai", name: "gpt-4o-mini", verified: true },
  { provider: "openai", name: "o3-mini", verified: false },
];

const createdConnection: ProviderConnection = {
  id: "conn-1",
  provider: "openai",
  label: null,
  baseUrl: null,
  apiMode: "auto",
  customHeaders: {},
  models: [],
  createdAt: 1700000000,
  lastValidatedAt: null,
  apiKeySet: true,
};

const createMock = vi.hoisted(() => vi.fn());
const updateMock = vi.hoisted(() => vi.fn());
const deleteMock = vi.hoisted(() => vi.fn());
const validateMock = vi.hoisted(() => vi.fn());
const createProfileMock = vi.hoisted(() => vi.fn());
const llmProfilesMock = vi.hoisted(() => vi.fn());

vi.mock("#/hooks/query/use-search-providers", () => ({
  useSearchProviders: () => ({ data: mockProviders, isLoading: false }),
}));

vi.mock("#/hooks/query/use-provider-models", () => ({
  useProviderModels: () => ({ data: mockCatalog, isLoading: false }),
}));

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => llmProfilesMock(),
}));

vi.mock("#/hooks/mutation/use-provider-connection-mutations", () => ({
  useCreateProviderConnection: () => ({
    mutateAsync: createMock,
    isPending: false,
  }),
  useUpdateProviderConnection: () => ({
    mutateAsync: updateMock,
    isPending: false,
  }),
  useDeleteProviderConnection: () => ({
    mutateAsync: deleteMock,
    isPending: false,
  }),
  useValidateProviderConnection: () => ({
    mutateAsync: validateMock,
    isPending: false,
  }),
  useCreateProfileFromConnection: () => ({
    mutateAsync: createProfileMock,
    isPending: false,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, opts?: Record<string, unknown>) => {
      if (typeof opts === "object" && opts !== null) {
        return Object.entries(opts).reduce(
          (acc, [k, v]) => acc.replace(`{{${k}}}`, String(v)),
          key,
        );
      }
      return key;
    },
    i18n: { language: "en" },
  }),
}));

function renderWithQuery(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
  );
}

describe("ConnectProviderWizard", () => {
  beforeEach(() => {
    createMock.mockReset();
    updateMock.mockReset();
    deleteMock.mockReset();
    validateMock.mockReset();
    createProfileMock.mockReset();
    llmProfilesMock.mockReset();
    deleteMock.mockResolvedValue(undefined);
    llmProfilesMock.mockReturnValue({
      data: { profiles: [], active_profile: null },
    });
    createProfileMock.mockResolvedValue({
      profileName: "x",
      model: "x",
      provider: "openai",
      connectionId: "conn-1",
    });
  });

  it("disables test until provider + key are entered", () => {
    renderWithQuery(<ConnectProviderWizard isOpen onClose={vi.fn()} />);
    expect(screen.getByTestId("connect-provider-test")).toBeDisabled();
  });

  it("creates a connection, validates, and advances to the model picker", async () => {
    const user = userEvent.setup();
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: true,
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"],
      error: null,
      validatedAt: 1700000100,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={vi.fn()}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-api-key"), "sk-test-key");
    await user.click(screen.getByTestId("connect-provider-test"));

    await waitFor(() => {
      expect(createMock).toHaveBeenCalledWith({
        provider: "openai",
        key: "sk-test-key",
        label: undefined,
        baseUrl: undefined,
        apiMode: "auto",
        customHeaders: {},
      });
    });
    expect(validateMock).toHaveBeenCalledWith("conn-1");
    // Advance to the pick step: validated summary + model list render.
    await waitFor(() =>
      expect(screen.getByTestId("connection-validated-summary")).toBeTruthy(),
    );
    expect(screen.getByTestId("connection-model-list-verified")).toBeTruthy();
  });

  it("pre-checks recommended models by default in the picker", async () => {
    const user = userEvent.setup();
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: true,
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"],
      error: null,
      validatedAt: 1700000100,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={vi.fn()}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-api-key"), "sk-test-key");
    await user.click(screen.getByTestId("connect-provider-test"));
    await waitFor(() =>
      expect(screen.getByTestId("connection-model-list-verified")).toBeTruthy(),
    );

    // Both recommended models are checked by default.
    expect(screen.getByTestId("connection-model-gpt-4o")).toBeChecked();
    expect(screen.getByTestId("connection-model-gpt-4o-mini")).toBeChecked();
    // The non-recommended model is behind the "More from" collapsible.
    expect(screen.queryByTestId("connection-model-o3-mini")).not.toBeTruthy();
    expect(screen.getByTestId("connection-more-toggle")).toBeTruthy();
  });

  it("keeps the selection cleared after the Clear bulk action", async () => {
    const user = userEvent.setup();
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: true,
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"],
      error: null,
      validatedAt: 1700000100,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={vi.fn()}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-api-key"), "sk-test-key");
    await user.click(screen.getByTestId("connect-provider-test"));
    await waitFor(() =>
      expect(screen.getByTestId("connection-model-gpt-4o")).toBeChecked(),
    );

    // Clearing must stick: the default-selection effect must not re-check the
    // recommended models when the selection becomes empty.
    await user.click(screen.getByTestId("connection-clear"));

    expect(screen.getByTestId("connection-model-gpt-4o")).not.toBeChecked();
    expect(
      screen.getByTestId("connection-model-gpt-4o-mini"),
    ).not.toBeChecked();
    // With nothing selected, save is disabled and the summary is hidden.
    expect(screen.getByTestId("connect-provider-save")).toBeDisabled();
    expect(screen.queryByTestId("connection-save-summary")).not.toBeTruthy();
  });

  it("caps the default recommended selection to remaining profile slots", async () => {
    const user = userEvent.setup();
    llmProfilesMock.mockReturnValue({
      data: {
        profiles: Array.from({ length: 49 }, (_, i) => ({
          name: `existing-${i}`,
          model: `provider/existing-${i}`,
        })),
        active_profile: null,
      },
    });
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: true,
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"],
      error: null,
      validatedAt: 1700000100,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={vi.fn()}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-api-key"), "sk-test-key");
    await user.click(screen.getByTestId("connect-provider-test"));
    await waitFor(() =>
      expect(screen.getByTestId("connection-model-gpt-4o")).toBeChecked(),
    );

    expect(
      screen.getByTestId("connection-model-gpt-4o-mini"),
    ).not.toBeChecked();
    expect(
      screen.getByTestId("connection-profile-limit-summary"),
    ).toHaveTextContent("SETTINGS$CONNECTION_PROFILE_LIMIT_SUMMARY");
  });

  it("prevents selecting more new models than remaining profile slots", async () => {
    const user = userEvent.setup();
    llmProfilesMock.mockReturnValue({
      data: {
        profiles: Array.from({ length: 49 }, (_, i) => ({
          name: `existing-${i}`,
          model: `provider/existing-${i}`,
        })),
        active_profile: null,
      },
    });
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: true,
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"],
      error: null,
      validatedAt: 1700000100,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={vi.fn()}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-api-key"), "sk-test-key");
    await user.click(screen.getByTestId("connect-provider-test"));
    await waitFor(() =>
      expect(screen.getByTestId("connection-model-gpt-4o")).toBeChecked(),
    );

    await user.click(screen.getByTestId("connection-model-gpt-4o-mini"));

    expect(
      screen.getByTestId("connection-model-gpt-4o-mini"),
    ).not.toBeChecked();
    expect(displayErrorToast).toHaveBeenCalledWith(
      "SETTINGS$CONNECTION_PROFILE_LIMIT_REACHED",
    );
  });

  it("creates one profile per selected model on save", async () => {
    const user = userEvent.setup();
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: true,
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"],
      error: null,
      validatedAt: 1700000100,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={vi.fn()}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-api-key"), "sk-test-key");
    await user.click(screen.getByTestId("connect-provider-test"));
    await waitFor(() =>
      expect(screen.getByTestId("connect-provider-save")).toBeTruthy(),
    );

    // Two recommended models are pre-checked; save creates two profiles.
    await user.click(screen.getByTestId("connect-provider-save"));

    await waitFor(() =>
      expect(screen.getByTestId("connection-done-summary")).toBeTruthy(),
    );
    expect(createProfileMock).toHaveBeenCalledTimes(2);
    expect(createProfileMock).toHaveBeenCalledWith({
      id: "conn-1",
      request: { profileName: "gpt-4o", model: "gpt-4o" },
    });
    expect(createProfileMock).toHaveBeenCalledWith({
      id: "conn-1",
      request: { profileName: "gpt-4o-mini", model: "gpt-4o-mini" },
    });
    // A saved connection is not cleaned up on close.
    expect(deleteMock).not.toHaveBeenCalled();
  });

  it("shows the invalid summary with Try again / Use a different key when validation fails", async () => {
    const user = userEvent.setup();
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: false,
      verified: true,
      models: [],
      error: "Invalid API key",
      validatedAt: null,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={vi.fn()}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-api-key"), "sk-bad");
    await user.click(screen.getByTestId("connect-provider-test"));

    await waitFor(() =>
      expect(screen.getByTestId("connection-invalid-summary")).toBeTruthy(),
    );
    expect(screen.getByTestId("connection-try-again")).toBeTruthy();
    expect(screen.getByTestId("connection-different-key")).toBeTruthy();
  });

  it("rotates the same connection on retry instead of creating a second one", async () => {
    const user = userEvent.setup();
    createMock.mockResolvedValue(createdConnection);
    validateMock
      .mockResolvedValueOnce({
        id: "conn-1",
        provider: "openai",
        ok: false,
        verified: true,
        models: [],
        error: "Invalid API key",
        validatedAt: null,
      })
      .mockResolvedValueOnce({
        id: "conn-1",
        provider: "openai",
        ok: true,
        verified: true,
        models: ["gpt-4o"],
        error: null,
        validatedAt: 1700000200,
      });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={vi.fn()}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-api-key"), "sk-bad");
    await user.click(screen.getByTestId("connect-provider-test"));
    await waitFor(() =>
      expect(screen.getByTestId("connection-invalid-summary")).toBeTruthy(),
    );

    // Retry with a corrected key via "Try again".
    await user.type(screen.getByTestId("connection-api-key"), "-fixed");
    await user.click(screen.getByTestId("connection-try-again"));

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    expect(createMock).toHaveBeenCalledTimes(1);
    expect(updateMock).toHaveBeenCalledWith({
      id: "conn-1",
      request: {
        key: "sk-bad-fixed",
        label: undefined,
        baseUrl: undefined,
        apiMode: "auto",
        customHeaders: {},
      },
    });
  });

  it("passes custom endpoint settings when creating the connection", async () => {
    const user = userEvent.setup();
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: true,
      models: ["gpt-4o"],
      error: null,
      validatedAt: 1700000100,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={vi.fn()}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-label"), "Proxy");
    await user.type(
      screen.getByTestId("connection-base-url"),
      "https://proxy.example/v1",
    );
    await user.click(screen.getByTestId("connection-api-mode"));
    await user.click(screen.getByText("Responses"));
    fireEvent.change(screen.getByTestId("connection-custom-headers"), {
      target: { value: '{"X-Org":"eng"}' },
    });
    await user.type(screen.getByTestId("connection-api-key"), "sk-test-key");
    await user.click(screen.getByTestId("connect-provider-test"));

    await waitFor(() => {
      expect(createMock).toHaveBeenCalledWith({
        provider: "openai",
        key: "sk-test-key",
        label: "Proxy",
        baseUrl: "https://proxy.example/v1",
        apiMode: "responses",
        customHeaders: { "X-Org": "eng" },
      });
    });
  });

  it("deletes the orphaned connection when cancelled after a failed validate", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: false,
      verified: true,
      models: [],
      error: "Invalid API key",
      validatedAt: null,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={onClose}
        defaultProvider="openai"
      />,
    );

    await user.type(screen.getByTestId("connection-api-key"), "sk-bad");
    await user.click(screen.getByTestId("connect-provider-test"));
    await waitFor(() =>
      expect(screen.getByTestId("connection-invalid-summary")).toBeTruthy(),
    );

    await user.click(screen.getByTestId("connect-provider-cancel"));
    expect(deleteMock).toHaveBeenCalledWith("conn-1");
    expect(onClose).toHaveBeenCalled();
  });
});
