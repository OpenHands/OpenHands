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

const openaiSubscriptionStatusMock = vi.hoisted(() =>
  vi.fn(() => ({ data: { connected: false } })),
);

vi.mock("#/hooks/query/use-llm-subscription-status", () => ({
  useOpenAISubscriptionStatus: () => openaiSubscriptionStatusMock(),
}));

const acpAuthStatusMock = vi.hoisted(() =>
  vi.fn((_providerKey?: unknown) => ({
    status: "unauthenticated" as
      | "authenticated"
      | "unauthenticated"
      | "unknown",
    isChecking: false,
    isSupported: true,
  })),
);

vi.mock("#/hooks/query/use-acp-auth-status", () => ({
  useAcpAuthStatus: (...args: unknown[]) => acpAuthStatusMock(...args),
}));

vi.mock("#/hooks/query/use-llm-subscription-models", () => ({
  useOpenAISubscriptionModels: () => ({ data: ["gpt-5.4"] }),
}));

const saveProfileMutate = vi.hoisted(() => vi.fn());

vi.mock("#/hooks/mutation/use-save-llm-profile", () => ({
  useSaveLlmProfile: () => ({
    mutateAsync: saveProfileMutate,
    isPending: false,
  }),
}));

vi.mock(
  "#/components/features/settings/llm-settings/openai-subscription-auth-card",
  () => ({
    OpenAISubscriptionAuthCard: () => (
      <div data-testid="openai-subscription-auth-card" />
    ),
  }),
);

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
    saveProfileMutate.mockReset();
    openaiSubscriptionStatusMock.mockReset();
    openaiSubscriptionStatusMock.mockReturnValue({
      data: { connected: false },
    });
    acpAuthStatusMock.mockReset();
    acpAuthStatusMock.mockReturnValue({
      status: "unauthenticated",
      isChecking: false,
      isSupported: true,
    });
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

  it("lists already-signed-in CLIs without adding a connection", () => {
    acpAuthStatusMock.mockImplementation((providerKey: unknown) => ({
      status:
        providerKey === "cursor-cli" ? "authenticated" : "unauthenticated",
      isChecking: false,
      isSupported: true,
    }));

    renderWith(
      <ProviderConnectionsManager
        connections={[]}
        linkedCountById={{}}
        isLoading={false}
        loadError={null}
      />,
    );

    expect(
      screen.queryByTestId("provider-connections-empty"),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Cursor CLI")).toBeInTheDocument();
    expect(screen.queryByText("OpenCode")).not.toBeInTheDocument();
    expect(
      screen.getByText(/Signed in|PROVIDER_CONNECTION_SIGNED_IN/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/0 model/)).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("provider-connection-delete"),
    ).not.toBeInTheDocument();
  });

  it("shows the detected model count on a signed-in CLI", () => {
    acpAuthStatusMock.mockImplementation((providerKey: unknown) => ({
      status:
        providerKey === "cursor-cli" ? "authenticated" : "unauthenticated",
      isChecking: false,
      isSupported: true,
    }));

    renderWith(
      <ProviderConnectionsManager
        connections={[]}
        linkedCountById={{}}
        catalogCountsBySource={{
          chatgpt: 0,
          claude: 0,
          "cursor-cli": 12,
          opencode: 0,
        }}
        isLoading={false}
        loadError={null}
      />,
    );

    expect(
      screen.getByText(/12 model|PROVIDER_CONNECTION_MODEL_COUNT/),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/Signed in|PROVIDER_CONNECTION_SIGNED_IN/),
    ).not.toBeInTheDocument();
  });

  it("lists an already-signed-in Claude subscription without adding a connection", () => {
    acpAuthStatusMock.mockImplementation((providerKey: unknown) => ({
      status:
        providerKey === "claude-code" ? "authenticated" : "unauthenticated",
      isChecking: false,
      isSupported: true,
    }));

    renderWith(
      <ProviderConnectionsManager
        connections={[]}
        linkedCountById={{}}
        isLoading={false}
        loadError={null}
      />,
    );

    expect(
      screen.queryByTestId("provider-connections-empty"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText(/Claude subscription|LLM_AUTH_TYPE_CLAUDE_SUBSCRIPTION/),
    ).toBeInTheDocument();
  });

  it("lists a ChatGPT subscription that is already connected on the host", () => {
    openaiSubscriptionStatusMock.mockReturnValue({ data: { connected: true } });

    renderWith(
      <ProviderConnectionsManager
        connections={[]}
        linkedCountById={{}}
        isLoading={false}
        loadError={null}
      />,
    );

    expect(
      screen.queryByTestId("provider-connections-empty"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText(/ChatGPT subscription|LLM_AUTH_TYPE_SUBSCRIPTION/),
    ).toBeInTheDocument();
  });

  it("shows providers a key or ChatGPT login can actually configure", async () => {
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
    expect(screen.getByText("Cursor CLI")).toBeInTheDocument();
    expect(screen.getByText("OpenCode")).toBeInTheDocument();
    expect(screen.getByText("OpenHands")).toBeInTheDocument();
    expect(screen.getByText("Ollama")).toBeInTheDocument();
    expect(screen.queryByText("Azure")).not.toBeInTheDocument();
    expect(screen.queryByText("chatgpt")).not.toBeInTheDocument();

    await user.click(screen.getByText("Anthropic"));
    expect(providerSelector).toHaveValue("Anthropic");
  });

  it("lets OpenAI connect with a ChatGPT subscription instead of an API key", async () => {
    const user = userEvent.setup();
    openaiSubscriptionStatusMock.mockReturnValue({ data: { connected: true } });
    saveProfileMutate.mockResolvedValue({});

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
      "ChatGPT",
    );

    const providerSelector = screen.getByRole("combobox", {
      name: /provider/i,
    });
    await user.click(providerSelector);
    await user.click(screen.getByTestId("provider-item-openai"));

    expect(
      screen.getByTestId("provider-connection-api-key-input"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("provider-connection-auth-input"));
    const chatgptAuthOptions = screen.getAllByText(
      /ChatGPT subscription|LLM_AUTH_TYPE_SUBSCRIPTION/,
    );
    await user.click(chatgptAuthOptions[chatgptAuthOptions.length - 1]);

    expect(
      screen.queryByTestId("provider-connection-api-key-input"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("openai-subscription-auth-card"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("provider-connection-submit"));

    await waitFor(() => {
      expect(saveProfileMutate).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "ChatGPT",
          request: expect.objectContaining({
            llm: expect.objectContaining({
              auth_type: "subscription",
              subscription_vendor: "openai",
              model: "gpt-5.4",
            }),
          }),
        }),
      );
    });
  });

  it("lets Anthropic connect with a Claude subscription instead of an API key", async () => {
    const user = userEvent.setup();
    acpAuthStatusMock.mockImplementation((providerKey: unknown) => ({
      status:
        providerKey === "claude-code" ? "authenticated" : "unauthenticated",
      isChecking: false,
      isSupported: true,
    }));
    const createSpy = vi.spyOn(ProviderConnectionsService, "create");

    renderWith(
      <ProviderConnectionsManager
        connections={[]}
        linkedCountById={{}}
        isLoading={false}
        loadError={null}
      />,
    );

    await user.click(screen.getByTestId("add-provider-connection"));
    await user.click(screen.getByRole("combobox", { name: /provider/i }));
    await user.click(screen.getByTestId("provider-item-anthropic"));

    expect(
      screen.getByTestId("provider-connection-api-key-input"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("provider-connection-auth-input"));
    const claudeAuthOptions = screen.getAllByText(
      /Claude subscription|LLM_AUTH_TYPE_CLAUDE_SUBSCRIPTION/,
    );
    await user.click(claudeAuthOptions[claudeAuthOptions.length - 1]);

    expect(
      screen.queryByTestId("provider-connection-api-key-input"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("cli-subscription-auth-card"),
    ).toBeInTheDocument();
    expect(acpAuthStatusMock).toHaveBeenCalledWith(
      "claude-code",
      expect.anything(),
    );

    await user.click(screen.getByTestId("provider-connection-submit"));

    await waitFor(() => {
      expect(displaySuccessToast).toHaveBeenCalled();
    });
    expect(createSpy).not.toHaveBeenCalled();
    expect(saveProfileMutate).not.toHaveBeenCalled();
  });

  it("lets Cursor CLI connect from a host login without an API key", async () => {
    const user = userEvent.setup();
    acpAuthStatusMock.mockImplementation((providerKey: unknown) => ({
      status:
        providerKey === "cursor-cli" ? "authenticated" : "unauthenticated",
      isChecking: false,
      isSupported: true,
    }));
    const createSpy = vi.spyOn(ProviderConnectionsService, "create");

    renderWith(
      <ProviderConnectionsManager
        connections={[]}
        linkedCountById={{}}
        isLoading={false}
        loadError={null}
      />,
    );

    await user.click(screen.getByTestId("add-provider-connection"));
    await user.click(screen.getByRole("combobox", { name: /provider/i }));
    await user.click(screen.getByTestId("provider-item-cursor-cli"));

    expect(
      screen.queryByTestId("provider-connection-auth-input"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("cli-subscription-auth-card"),
    ).toBeInTheDocument();
    expect(acpAuthStatusMock).toHaveBeenCalledWith(
      "cursor-cli",
      expect.anything(),
    );

    await user.click(screen.getByTestId("provider-connection-submit"));

    await waitFor(() => {
      expect(displaySuccessToast).toHaveBeenCalled();
    });
    expect(createSpy).not.toHaveBeenCalled();
  });

  it("requires a base URL for Ollama and not an API key", async () => {
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
    await user.type(
      screen.getByTestId("provider-connection-name-input"),
      "Local",
    );

    const providerSelector = screen.getByRole("combobox", {
      name: /provider/i,
    });
    await user.click(providerSelector);
    await user.click(screen.getByTestId("provider-item-ollama"));

    expect(screen.getByTestId("provider-connection-submit")).toBeDisabled();

    await user.type(
      screen.getByTestId("provider-connection-base-url-input"),
      "http://127.0.0.1:11434",
    );
    expect(screen.getByTestId("provider-connection-submit")).toBeEnabled();
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
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(
      screen.getByText(/3 profile|PROVIDER_CONNECTION_PROFILE_COUNT/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/3 model/)).not.toBeInTheDocument();
  });

  it("shows supported providers in the edit-connection selector", async () => {
    const user = userEvent.setup();

    renderWith(
      <ProviderConnectionsManager
        connections={[connection]}
        linkedCountById={{}}
        isLoading={false}
        loadError={null}
      />,
    );

    await user.click(screen.getByTestId("provider-connection-edit"));

    const providerSelector = screen.getByRole("combobox", {
      name: /provider/i,
    });
    expect(providerSelector).toHaveValue("OpenAI");

    await user.click(providerSelector);

    expect(screen.getByText("Anthropic")).toBeInTheDocument();
    expect(screen.getByText("OpenHands")).toBeInTheDocument();
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
