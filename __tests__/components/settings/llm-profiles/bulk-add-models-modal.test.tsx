import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BulkAddModelsModal } from "#/components/features/settings/llm-profiles/bulk-add-models-modal";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string | number>) => {
      const translations: Record<string, string> = {
        SETTINGS$PROVIDER_CONNECTION_BULK_ADD_TITLE: "Bulk add models",
        SETTINGS$PROVIDER_CONNECTION_BULK_ADD_TEXTAREA_LABEL: "Model IDs",
        SETTINGS$PROVIDER_CONNECTION_BULK_ADD_HINT: "Paste model IDs.",
        SETTINGS$PROVIDER_CONNECTION_BULK_ADD_PREVIEW:
          "{{create}} new, {{skip}} already exist",
        SETTINGS$PROVIDER_CONNECTION_BULK_ADD_SUBMIT:
          "Create {{count}} profile(s)",
        SETTINGS$PROVIDER_CONNECTION_BULK_ADD_RESULT_CREATED:
          "Created {{count}} profile(s)",
        SETTINGS$PROVIDER_CONNECTION_BULK_ADD_RESULT_FAILED:
          "Failed: {{names}}",
        BUTTON$CANCEL: "Cancel",
        BUTTON$CLOSE: "Close",
        ERROR$GENERIC: "An error occurred",
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

vi.mock("#/api/profiles-service/profiles-service.api");

const connection: ProviderConnection = {
  id: "conn-1",
  display_name: "Ollama Cloud",
  provider: "openai",
  base_url: "https://ollama.com/v1",
  created_at: 1,
  updated_at: 2,
  api_key_set: true,
};

describe("BulkAddModelsModal", () => {
  let queryClient: QueryClient;

  const renderModal = (
    props: Partial<Parameters<typeof BulkAddModelsModal>[0]> = {},
  ) =>
    render(
      <QueryClientProvider client={queryClient}>
        <BulkAddModelsModal
          connection={connection}
          existingProfileNames={new Set()}
          onClose={vi.fn()}
          {...props}
        />
      </QueryClientProvider>,
    );

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

  it("returns null when connection is null", () => {
    const { container } = render(
      <QueryClientProvider client={queryClient}>
        <BulkAddModelsModal
          connection={null}
          existingProfileNames={new Set()}
          onClose={vi.fn()}
        />
      </QueryClientProvider>,
    );
    expect(container.firstChild).toBeNull();
  });

  it("parses newline- and comma-separated model ids and previews new vs skipped", async () => {
    const user = userEvent.setup();
    renderModal({ existingProfileNames: new Set(["gpt-oss-120b-cloud"]) });

    await user.type(
      screen.getByTestId("bulk-add-models-textarea"),
      "gpt-oss:120b-cloud,glm-5.3-flash:cloud\nglm-5.3-flash:cloud\nkimi-k3",
    );

    // gpt-oss:120b-cloud -> already exists (skip), glm-5.3-flash:cloud
    // appears twice (deduped to one), kimi-k3 is new -> 2 create, 1 skip
    expect(screen.getByTestId("bulk-add-models-preview")).toHaveTextContent(
      "2 new, 1 already exist",
    );
  });

  it("creates one profile per model id, linked to the connection", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue({
      name: "x",
      message: "ok",
    });
    renderModal();

    await user.type(
      screen.getByTestId("bulk-add-models-textarea"),
      "gpt-oss:120b-cloud\nglm-5.3-flash:cloud",
    );
    await user.click(screen.getByTestId("bulk-add-models-submit"));

    await waitFor(() => {
      expect(ProfilesService.saveProfile).toHaveBeenCalledTimes(2);
    });

    expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
      "gpt-oss-120b-cloud",
      {
        llm: {
          model: "openai/gpt-oss:120b-cloud",
          provider_connection_id: "conn-1",
          auth_type: "api_key",
          subscription_vendor: null,
        },
        include_secrets: true,
      },
    );
    expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
      "glm-5.3-flash-cloud",
      {
        llm: {
          model: "openai/glm-5.3-flash:cloud",
          provider_connection_id: "conn-1",
          auth_type: "api_key",
          subscription_vendor: null,
        },
        include_secrets: true,
      },
    );

    await screen.findByTestId("bulk-add-models-result-created");
    expect(
      screen.getByTestId("bulk-add-models-result-created"),
    ).toHaveTextContent("Created 2 profile(s)");
  });

  it("uses a model id verbatim when it already carries its own provider prefix", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.saveProfile).mockResolvedValue({
      name: "x",
      message: "ok",
    });
    renderModal();

    await user.type(
      screen.getByTestId("bulk-add-models-textarea"),
      "anthropic/claude-x",
    );
    await user.click(screen.getByTestId("bulk-add-models-submit"));

    await waitFor(() => {
      expect(ProfilesService.saveProfile).toHaveBeenCalledWith(
        "claude-x",
        expect.objectContaining({
          llm: expect.objectContaining({ model: "anthropic/claude-x" }),
        }),
      );
    });
  });

  it("reports per-model failures without blocking the others", async () => {
    const user = userEvent.setup();
    vi.mocked(ProfilesService.saveProfile).mockImplementation(
      async (name: string) => {
        if (name === "bad-model") throw new Error("model not found");
        return { name, message: "ok" };
      },
    );
    renderModal();

    await user.type(
      screen.getByTestId("bulk-add-models-textarea"),
      "good-model\nbad-model",
    );
    await user.click(screen.getByTestId("bulk-add-models-submit"));

    await screen.findByTestId("bulk-add-models-result-created");
    expect(
      screen.getByTestId("bulk-add-models-result-created"),
    ).toHaveTextContent("Created 1 profile(s)");
    expect(
      screen.getByTestId("bulk-add-models-result-failed"),
    ).toHaveTextContent("Failed: bad-model");
  });

  it("disables submit until at least one new model is parsed", () => {
    renderModal();
    expect(screen.getByTestId("bulk-add-models-submit")).toBeDisabled();
  });
});
