import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConnectProviderWizard } from "#/components/features/settings/provider-connections/connect-provider-wizard";
import type { ProviderConnection } from "#/api/provider-connections-service";

const mockProviders = [{ name: "openai", verified: true }];

const createdConnection: ProviderConnection = {
  id: "conn-1",
  provider: "openai",
  label: null,
  models: [],
  createdAt: 1700000000,
  lastValidatedAt: null,
  apiKeySet: true,
};

const createMock = vi.hoisted(() => vi.fn());
const updateMock = vi.hoisted(() => vi.fn());
const deleteMock = vi.hoisted(() => vi.fn());
const validateMock = vi.hoisted(() => vi.fn());

vi.mock("#/hooks/query/use-search-providers", () => ({
  useSearchProviders: () => ({ data: mockProviders, isLoading: false }),
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
    mutate: deleteMock,
    isPending: false,
  }),
  useValidateProviderConnection: () => ({
    mutateAsync: validateMock,
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
  });

  it("disables submit until provider + key are entered", () => {
    renderWithQuery(<ConnectProviderWizard isOpen onClose={vi.fn()} />);
    const submit = screen.getByTestId("connect-provider-submit");
    expect(submit).toBeDisabled();
  });

  it("creates a connection and validates it on submit", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    createMock.mockResolvedValue(createdConnection);
    validateMock.mockResolvedValue({
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: false,
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"],
      error: null,
      validatedAt: 1700000100,
    });

    renderWithQuery(
      <ConnectProviderWizard
        isOpen
        onClose={onClose}
        defaultProvider="openai"
      />,
    );

    // Provider is preselected; enter a key.
    await user.type(screen.getByTestId("connection-api-key"), "sk-test-key");
    await user.click(screen.getByTestId("connect-provider-submit"));

    await waitFor(() => {
      expect(createMock).toHaveBeenCalledWith({
        provider: "openai",
        key: "sk-test-key",
        label: undefined,
      });
    });
    expect(validateMock).toHaveBeenCalledWith("conn-1");
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    // A committed connection is not cleaned up.
    expect(deleteMock).not.toHaveBeenCalled();
  });

  it("shows the invalid summary when validation fails", async () => {
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
    await user.click(screen.getByTestId("connect-provider-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("connection-invalid-summary")).toBeTruthy(),
    );
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
    await user.click(screen.getByTestId("connect-provider-submit"));
    await waitFor(() =>
      expect(screen.getByTestId("connection-invalid-summary")).toBeTruthy(),
    );

    // Retry with a corrected key.
    await user.type(screen.getByTestId("connection-api-key"), "-fixed");
    await user.click(screen.getByTestId("connect-provider-submit"));

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    // Only one create ever happened; the retry rotated the existing record.
    expect(createMock).toHaveBeenCalledTimes(1);
    expect(updateMock).toHaveBeenCalledWith({
      id: "conn-1",
      request: { key: "sk-bad-fixed", label: undefined },
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
    await user.click(screen.getByTestId("connect-provider-submit"));
    await waitFor(() =>
      expect(screen.getByTestId("connection-invalid-summary")).toBeTruthy(),
    );

    await user.click(screen.getByTestId("connect-provider-cancel"));
    expect(deleteMock).toHaveBeenCalledWith("conn-1");
    expect(onClose).toHaveBeenCalled();
  });
});
