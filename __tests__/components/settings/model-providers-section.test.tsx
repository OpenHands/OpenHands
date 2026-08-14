import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ModelProvidersSection } from "#/components/features/settings/model-providers/model-providers-section";
import type { ModelProvider } from "#/api/model-providers-service";

const provider: ModelProvider = {
  id: "prov-openai",
  kind: "openai",
  displayName: "OpenAI",
  baseUrl: "https://api.openai.com/v1",
  wireApi: "auto",
  customHeaders: {},
  models: [
    { name: "gpt-5.6-luna", wireApi: null },
    { name: "gpt-5.6-sol", wireApi: null },
    { name: "gpt-5.6-terra", wireApi: null },
  ],
  createdAt: 1700000000,
  updatedAt: 1700000100,
  apiKeySet: true,
};

const addModelMock = vi.hoisted(() => vi.fn());
const removeModelMock = vi.hoisted(() => vi.fn());
const deleteProviderMock = vi.hoisted(() => vi.fn());

vi.mock("#/hooks/query/use-model-providers", () => ({
  useModelProviders: () => ({
    data: [provider],
    isLoading: false,
    error: null,
  }),
}));

vi.mock("#/hooks/use-can-manage-org-profiles", () => ({
  useCanManageOrgProfiles: () => true,
}));

// Keep the modal form inert; this suite exercises the section + cards.
vi.mock("#/components/features/settings/model-providers/provider-form", () => ({
  ProviderForm: () => null,
}));

vi.mock("#/hooks/mutation/use-model-provider-mutations", () => ({
  useCreateModelProvider: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUpdateModelProvider: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useDeleteModelProvider: () => ({
    mutateAsync: deleteProviderMock,
    isPending: false,
  }),
  useAddProviderModel: () => ({ mutateAsync: addModelMock, isPending: false }),
  useUpdateProviderModel: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useRemoveProviderModel: () => ({
    mutateAsync: removeModelMock,
    isPending: false,
  }),
  useTestModelProvider: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

// The service throws on cloud; provide a local (no-op) assertion so the
// section renders its normal state.
vi.mock("#/api/model-providers-service", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/model-providers-service")>();
  return {
    ...actual,
    assertProvidersSupportedLocally: () => undefined,
    isModelProvidersNotOnCloudError: () => false,
  };
});

function renderSection() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ModelProvidersSection />
    </QueryClientProvider>,
  );
}

describe("ModelProvidersSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the managed GitHub Copilot card and configured providers", () => {
    renderSection();
    expect(
      screen.getByTestId("managed-provider-github-copilot"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("provider-card-prov-openai")).toBeInTheDocument();
    // Header shows the display name; all three nested models render.
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByTestId("model-row-gpt-5.6-luna")).toBeInTheDocument();
    expect(screen.getByTestId("model-row-gpt-5.6-sol")).toBeInTheDocument();
    expect(screen.getByTestId("model-row-gpt-5.6-terra")).toBeInTheDocument();
  });

  it("opens the add-provider preset menu", async () => {
    const user = userEvent.setup();
    renderSection();
    await user.click(screen.getByTestId("add-provider-button"));
    expect(screen.getByTestId("add-provider-menu")).toBeInTheDocument();
    expect(
      screen.getByTestId("add-provider-option-anthropic"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("add-provider-option-custom"),
    ).toBeInTheDocument();
  });

  it("adds a model under the provider", async () => {
    addModelMock.mockResolvedValue(provider);
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByTestId("provider-add-model-prov-openai"));
    await user.type(
      screen.getByTestId("add-model-input-prov-openai"),
      "gpt-5.6-nova",
    );
    await user.click(screen.getByTestId("add-model-confirm-prov-openai"));

    await waitFor(() =>
      expect(addModelMock).toHaveBeenCalledWith({
        id: "prov-openai",
        model: { name: "gpt-5.6-nova" },
      }),
    );
  });

  it("removes a model from the provider", async () => {
    removeModelMock.mockResolvedValue(provider);
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByTestId("model-remove-gpt-5.6-sol"));

    await waitFor(() =>
      expect(removeModelMock).toHaveBeenCalledWith({
        id: "prov-openai",
        modelName: "gpt-5.6-sol",
      }),
    );
  });

  it("deletes a provider after confirmation", async () => {
    deleteProviderMock.mockResolvedValue(provider);
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByTestId("provider-delete-prov-openai"));
    await user.click(screen.getByTestId("provider-delete-confirm-prov-openai"));

    await waitFor(() =>
      expect(deleteProviderMock).toHaveBeenCalledWith("prov-openai"),
    );
  });
});
