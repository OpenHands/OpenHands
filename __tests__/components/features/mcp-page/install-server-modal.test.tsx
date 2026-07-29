import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import McpService from "#/api/mcp-service/mcp-service.api";
import {
  __resetMcpHealthStoreForTests,
  getMcpHealthSnapshot,
} from "#/api/mcp-health/mcp-health-store";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";
import { SecretsService } from "#/api/secrets-service";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { InstallServerModal } from "#/components/features/mcp-page/install-server-modal";
import {
  INTEGRATION_CATALOG as MCP_MARKETPLACE,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import { getMcpMarketplaceCatalog } from "#/utils/mcp-marketplace-utils";

function renderWith(ui: React.ReactNode) {
  return render(ui, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });
}

describe("InstallServerModal", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
    // Default: pre-flight test passes so existing save tests remain unaffected.
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: true,
      tools: [],
    });
  });

  it("uses Slack's API fallback when the default option is OAuth", async () => {
    const slack = MCP_MARKETPLACE.find((e) => e.id === "slack")!;
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    const onClose = vi.fn();
    renderWith(<InstallServerModal existingServers={[]} entry={slack} onClose={onClose} />);

    await screen.findByTestId("mcp-install-modal");

    // Fail fast when required fields are empty.
    fireEvent.click(screen.getByTestId("mcp-install-submit"));
    await waitFor(() => {
      expect(saveSpy).not.toHaveBeenCalled();
    });

    fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_BOT_TOKEN"), {
      target: { value: "xoxb-abc" },
    });
    fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_TEAM_ID"), {
      target: { value: "T01" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    const [payload] = saveSpy.mock.calls[0];
    const sentMcpConfig = (payload as Record<string, unknown>)
      .agent_settings_diff as {
      mcp_config: Record<string, unknown>;
    };
    expect(sentMcpConfig.mcp_config).toMatchObject({
      slack: {
        command: "npx",
        args: ["-y", "@zencoderai/slack-mcp-server"],
        env: { SLACK_BOT_TOKEN: "xoxb-abc", SLACK_TEAM_ID: "T01" },
      },
    });
    expect(onClose).toHaveBeenCalled();
  });

  it("seeds the new card's health from the pre-save connection test", async () => {
    // Arrange: a fresh install whose pre-save test passes. Its card must
    // show a verdict immediately after install, without a second probe.
    __resetMcpHealthStoreForTests();
    const slack = MCP_MARKETPLACE.find((e) => e.id === "slack")!;
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);

    renderWith(
      <InstallServerModal existingServers={[]} entry={slack} onClose={vi.fn()} />,
    );
    await screen.findByTestId("mcp-install-modal");

    // Act: complete the install.
    fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_BOT_TOKEN"), {
      target: { value: "xoxb-abc" },
    });
    fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_TEAM_ID"), {
      target: { value: "T01" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    // Assert: the health store carries the saved config's verdict (the
    // mocked test returns no tool_result, so it proves connectivity only).
    const expectedKey = getMcpServerHealthKey({
      id: "irrelevant-for-key",
      type: "stdio",
      name: "slack",
      command: "npx",
      args: ["-y", "@zencoderai/slack-mcp-server"],
      env: { SLACK_TEAM_ID: "T01", SLACK_BOT_TOKEN: "xoxb-abc" },
    });
    await waitFor(() =>
      expect(getMcpHealthSnapshot()[expectedKey]).toMatchObject({
        status: "healthy",
        verification: "connectivity-only",
      }),
    );
  });

  it("installs Tavily as a stdio MCP server with TAVILY_API_KEY env", async () => {
    // Tavily was previously a fake `kind: "tavily-builtin"` template
    // that called saveSettings({ search_api_key }) — but that field
    // was dropped on the floor in both local and cloud save paths, so
    // installing Tavily silently did nothing. It's now a regular
    // stdio MCP entry (`npx -y tavily-mcp` + TAVILY_API_KEY) that
    // goes through the same mcp_config write as every other entry.
    const tavily = MCP_MARKETPLACE.find((e) => e.id === "tavily")!;
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    const onClose = vi.fn();
    renderWith(<InstallServerModal existingServers={[]} entry={tavily} onClose={onClose} />);

    await screen.findByTestId("mcp-install-modal");

    // Submit with no key fails the required-field check.
    fireEvent.click(screen.getByTestId("mcp-install-submit"));
    await waitFor(() => expect(saveSpy).not.toHaveBeenCalled());

    fireEvent.change(screen.getByTestId("mcp-install-field-TAVILY_API_KEY"), {
      target: { value: "tvly-secret" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    const sent = (saveSpy.mock.calls[0][0] as Record<string, unknown>)
      .agent_settings_diff as {
      mcp_config: Record<string, unknown>;
    };
    expect(sent.mcp_config).toMatchObject({
      tavily: {
        command: "npx",
        args: ["-y", "tavily-mcp"],
        env: { TAVILY_API_KEY: "tvly-secret" },
      },
    });
    expect(onClose).toHaveBeenCalled();
  });

  it("blocks submission of an shttp template when api_key is required and empty", async () => {
    // Build a synthetic catalog entry with apiKeyOptional: false so we
    // exercise the new required-key validation in handleHttpServerSubmit
    // without relying on the catalog choosing to mark one this way.
    const entry: MarketplaceEntry = {
      id: "synthetic-required",
      name: "Synthetic",
      description: "Synthetic catalog entry used in tests.",
      docsUrl: "https://example.com/docs",
      iconBg: "#000000",
      connectionOptions: [
        {
          id: "api",
          provider: "mcp",
          transport: {
            kind: "shttp",
            url: "https://example.com/mcp",
            apiKeyOptional: false,
          },
          auth: { strategy: "api_key" },
        },
      ],
    };
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderWith(<InstallServerModal existingServers={[]} entry={entry} onClose={vi.fn()} />);

    await screen.findByTestId("mcp-install-modal");

    fireEvent.click(screen.getByTestId("mcp-install-submit"));
    // No save call until the user fills in the key.
    await waitFor(() => {
      expect(saveSpy).not.toHaveBeenCalled();
    });

    fireEvent.change(screen.getByTestId("mcp-install-field-api_key"), {
      target: { value: "secret-123" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
  });

  it("allows submitting an shttp template with no key when apiKeyOptional is true", async () => {
    const entry: MarketplaceEntry = {
      id: "synthetic-optional",
      name: "Synthetic Optional",
      description: "Synthetic entry that allows empty api_key.",
      docsUrl: "https://example.com/docs",
      iconBg: "#000000",
      connectionOptions: [
        {
          id: "api",
          provider: "mcp",
          transport: {
            kind: "shttp",
            url: "https://example.com/mcp",
            apiKeyOptional: true,
          },
          auth: { strategy: "api_key", apiKeyOptional: true },
        },
      ],
    };
    const getSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderWith(<InstallServerModal existingServers={[]} entry={entry} onClose={vi.fn()} />);

    await screen.findByTestId("mcp-install-modal");
    // The add-mcp-server mutation bails when useSettings() hasn't
    // resolved yet, so wait for the initial settings fetch before
    // submitting — otherwise the test races React Query.
    await waitFor(() => expect(getSpy).toHaveBeenCalled());
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
  });

  it("persists OAuth state returned by the connection test when installing", async () => {
    const entry: MarketplaceEntry = {
      id: "synthetic-oauth",
      name: "Synthetic OAuth",
      description: "Synthetic OAuth entry.",
      docsUrl: "https://example.com/docs",
      iconBg: "#000000",
      connectionOptions: [
        {
          id: "oauth",
          provider: "mcp",
          transport: {
            kind: "shttp",
            url: "https://mcp.example.com/mcp",
          },
          auth: {
            strategy: "oauth2",
            oauth: { clientAuthentication: "none" },
          },
        },
      ],
    };
    vi.spyOn(McpService, "authorizeOAuth").mockResolvedValue({
      ok: true,
      tools: [],
      oauth_state: {
        tokens: { access_token: "gAAAAencrypted-access-token" },
        token_expires_at: 12345,
      },
    });
    const getSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderWith(<InstallServerModal existingServers={[]} entry={entry} onClose={vi.fn()} />);
    await screen.findByTestId("mcp-install-modal");
    await waitFor(() => expect(getSpy).toHaveBeenCalled());
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    const sent = (saveSpy.mock.calls[0][0] as Record<string, unknown>)
      .agent_settings_diff as {
      mcp_config: Record<string, unknown>;
    };
    expect(sent.mcp_config).toMatchObject({
      "synthetic-oauth": {
        url: "https://mcp.example.com/mcp",
        auth: {
          strategy: "oauth2",
          state: {
            tokens: { access_token: "gAAAAencrypted-access-token" },
            token_expires_at: 12345,
          },
        },
      },
    });
  });

  it("installs header-field remote servers with tagged header auth", async () => {
    const entry = {
      id: "datadog-style",
      name: "Datadog-style Server",
      description: "Remote MCP server that authenticates via two headers.",
      iconBg: "#632CA6",
      connectionOptions: [
        {
          id: "api",
          provider: "mcp",
          transport: {
            kind: "shttp",
            url: "https://mcp.example.com/mcp",
            headerFields: [
              {
                key: "DD-API-KEY",
                label: "Datadog API key",
                type: "password",
                required: true,
              },
              {
                key: "DD-APPLICATION-KEY",
                label: "Datadog Application key",
                type: "password",
                required: true,
              },
            ],
          },
          auth: { strategy: "none" },
        },
      ],
    } as unknown as MarketplaceEntry;
    const testSpy = vi
      .spyOn(McpService, "testServer")
      .mockResolvedValue({ ok: true, tools: [] });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderWith(<InstallServerModal existingServers={[]} entry={entry} onClose={vi.fn()} />);
    await screen.findByTestId("mcp-install-modal");
    await waitFor(() => expect(SettingsService.getSettings).toHaveBeenCalled());

    expect(
      screen.queryByTestId("mcp-install-field-api_key"),
    ).not.toBeInTheDocument();
    fireEvent.change(screen.getByTestId("mcp-install-field-DD-API-KEY"), {
      target: { value: "dd-api-secret" },
    });
    fireEvent.change(
      screen.getByTestId("mcp-install-field-DD-APPLICATION-KEY"),
      { target: { value: "dd-app-secret" } },
    );
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    expect(testSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "shttp",
        url: "https://mcp.example.com/mcp",
        auth: {
          strategy: "header",
          headers: {
            "DD-API-KEY": "dd-api-secret",
            "DD-APPLICATION-KEY": "dd-app-secret",
          },
        },
      }),
    );
    const sent = (saveSpy.mock.calls[0][0] as Record<string, unknown>)
      .agent_settings_diff as {
      mcp_config: Record<string, unknown>;
    };
    expect(sent.mcp_config).toMatchObject({
      "datadog-style": {
        url: "https://mcp.example.com/mcp",
        auth: {
          strategy: "header",
          headers: {
            "DD-API-KEY": "dd-api-secret",
            "DD-APPLICATION-KEY": "dd-app-secret",
          },
        },
      },
    });
  });

  it("uses the user-edited URL when the transport opts into urlEditable", async () => {
    const entry = {
      id: "datadog-style",
      name: "Datadog-style Server",
      description: "Remote MCP server with a site-specific URL.",
      iconBg: "#632CA6",
      connectionOptions: [
        {
          id: "api",
          provider: "mcp",
          transport: {
            kind: "shttp",
            url: "https://mcp.example.com/mcp",
            urlEditable: true,
            headerFields: [
              {
                key: "DD-API-KEY",
                label: "Datadog API key",
                type: "password",
                required: true,
              },
            ],
          },
          auth: { strategy: "none" },
        },
      ],
    } as unknown as MarketplaceEntry;
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderWith(<InstallServerModal existingServers={[]} entry={entry} onClose={vi.fn()} />);
    await screen.findByTestId("mcp-install-modal");
    await waitFor(() => expect(SettingsService.getSettings).toHaveBeenCalled());

    const urlInput = screen.getByTestId(
      "mcp-install-field-url",
    ) as HTMLInputElement;
    expect(urlInput).not.toBeDisabled();
    fireEvent.change(urlInput, {
      target: { value: "https://mcp.us5.example.com/v1/mcp" },
    });
    fireEvent.change(screen.getByTestId("mcp-install-field-DD-API-KEY"), {
      target: { value: "dd-api-secret" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    const sent = (saveSpy.mock.calls[0][0] as Record<string, unknown>)
      .agent_settings_diff as {
      mcp_config: Record<string, unknown>;
    };
    expect(sent.mcp_config).toMatchObject({
      "datadog-style": {
        url: "https://mcp.us5.example.com/v1/mcp",
        auth: {
          strategy: "header",
          headers: { "DD-API-KEY": "dd-api-secret" },
        },
      },
    });
  });

  it("installs Linear over streamable HTTP with the api key as a bearer credential", async () => {
    // Arrange: the marketplace serves the patched Linear entry (shttp
    // /mcp endpoint, bearer auth) — the UI must never touch the removed
    // /sse transport.
    const linear = getMcpMarketplaceCatalog(MCP_MARKETPLACE).find(
      (e) => e.id === "linear",
    )!;
    const testSpy = vi
      .spyOn(McpService, "testServer")
      .mockResolvedValue({ ok: true, tools: [] });
    const getSpy = vi
      .spyOn(SettingsService, "getSettings")
      .mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderWith(<InstallServerModal existingServers={[]} entry={linear} onClose={vi.fn()} />);
    await screen.findByTestId("mcp-install-modal");
    // Wait for useSettings() so the add-mcp-server mutation doesn't bail.
    await waitFor(() => expect(getSpy).toHaveBeenCalled());

    // Act: provide the optional Linear API key and install.
    fireEvent.change(screen.getByTestId("mcp-install-field-api_key"), {
      target: { value: "lin_api_secret" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    // Assert: both the pre-flight test and the persisted config target
    // the new endpoint over streamable HTTP with the bearer credential.
    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    expect(testSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "shttp",
        url: "https://mcp.linear.app/mcp",
        auth: { strategy: "bearer", value: "lin_api_secret" },
      }),
    );
    const sent = (saveSpy.mock.calls[0][0] as Record<string, unknown>)
      .agent_settings_diff as {
      mcp_config: Record<string, unknown>;
    };
    // Remote installs are now keyed by the catalog slug ("linear") rather
    // than the auto-generated "shttp" fallback, so the server is
    // referenceable by name in mcp_server_refs.
    expect(sent.mcp_config).toMatchObject({
      linear: {
        url: "https://mcp.linear.app/mcp",
        auth: { strategy: "bearer", value: "lin_api_secret" },
      },
    });
  });

  it("closes from the top-right close button", async () => {
    const onClose = vi.fn();
    const slack = MCP_MARKETPLACE.find((e) => e.id === "slack")!;
    renderWith(<InstallServerModal existingServers={[]} entry={slack} onClose={onClose} />);
    await screen.findByTestId("mcp-install-modal");

    fireEvent.click(screen.getByTestId("mcp-install-modal-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("places Cancel before Install in the footer so the dominant action is the last focusable button", async () => {
    // Arrange: render with any marketplace entry so the footer is mounted.
    const slack = MCP_MARKETPLACE.find((e) => e.id === "slack")!;
    renderWith(<InstallServerModal existingServers={[]} entry={slack} onClose={vi.fn()} />);
    await screen.findByTestId("mcp-install-modal");

    // Act: locate both footer buttons.
    const cancel = screen.getByTestId("mcp-install-cancel");
    const submit = screen.getByTestId("mcp-install-submit");

    // Assert: Cancel precedes the dominant Install action in DOM order.
    expect(
      cancel.compareDocumentPosition(submit) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("shows an inline error, does not save, and keeps the modal open when the pre-flight test fails", async () => {
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: false,
      error: "ECONNREFUSED",
      error_kind: "connection",
    });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const onClose = vi.fn();

    const entry: MarketplaceEntry = {
      id: "synthetic-test-fail",
      name: "Failing Server",
      description: "Always fails the connection test.",
      docsUrl: "https://example.com/docs",
      iconBg: "#000000",
      connectionOptions: [
        {
          id: "api",
          provider: "mcp",
          transport: {
            kind: "shttp",
            url: "https://example.com/mcp",
            apiKeyOptional: true,
          },
          auth: { strategy: "api_key", apiKeyOptional: true },
        },
      ],
    };

    renderWith(<InstallServerModal existingServers={[]} entry={entry} onClose={onClose} />);
    await screen.findByTestId("mcp-install-modal");

    // Wait for settings to load so the mutation isn't a no-op.
    await waitFor(() => expect(SettingsService.getSettings).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    // Error message must appear.
    await waitFor(() =>
      expect(screen.getByTestId("mcp-install-modal-error")).toBeInTheDocument(),
    );

    // Save must never have been called.
    expect(saveSpy).not.toHaveBeenCalled();

    // Modal must stay open.
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByTestId("mcp-install-modal")).toBeInTheDocument();
  });

  it("shows the credential-specific message when the pre-flight test reports invalid credentials", async () => {
    // Arrange: installing Slack with credentials its verification call
    // rejects (the service maps Slack's invalid_auth to "credentials").
    const slack = MCP_MARKETPLACE.find((e) => e.id === "slack")!;
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: false,
      error: "invalid_auth",
      error_kind: "credentials",
    });

    renderWith(<InstallServerModal existingServers={[]} entry={slack} onClose={vi.fn()} />);
    await screen.findByTestId("mcp-install-modal");

    // Act: fill the required fields and install.
    fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_BOT_TOKEN"), {
      target: { value: "xoxb-invalid" },
    });
    fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_TEAM_ID"), {
      target: { value: "T-INVALID" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    // Assert: the credentials message is rendered (i18n keys are returned
    // as-is in tests), not the generic connection/unknown wording.
    await waitFor(() =>
      expect(screen.getByTestId("mcp-install-modal-error")).toHaveTextContent(
        "MCP$TEST_ERROR_CREDENTIALS",
      ),
    );
  });

  it("calls save and closes the modal when the pre-flight test succeeds", async () => {
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: true,
      tools: ["tool_a"],
    });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const onClose = vi.fn();

    const entry: MarketplaceEntry = {
      id: "synthetic-test-pass",
      name: "Passing Server",
      description: "Always passes the connection test.",
      docsUrl: "https://example.com/docs",
      iconBg: "#000000",
      connectionOptions: [
        {
          id: "api",
          provider: "mcp",
          transport: {
            kind: "shttp",
            url: "https://example.com/mcp",
            apiKeyOptional: true,
          },
          auth: { strategy: "api_key", apiKeyOptional: true },
        },
      ],
    };

    renderWith(<InstallServerModal existingServers={[]} entry={entry} onClose={onClose} />);
    await screen.findByTestId("mcp-install-modal");

    await waitFor(() => expect(SettingsService.getSettings).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(
      screen.queryByTestId("mcp-install-modal-error"),
    ).not.toBeInTheDocument();
  });

  it("shows Verifying… on the install button while the pre-flight test is in flight", async () => {
    // Never resolve so the test stays pending long enough to observe the label.
    vi.spyOn(McpService, "testServer").mockImplementation(
      () => new Promise(() => {}),
    );

    const entry: MarketplaceEntry = {
      id: "synthetic-pending",
      name: "Pending Server",
      description: "Connection test never resolves.",
      docsUrl: "https://example.com/docs",
      iconBg: "#000000",
      connectionOptions: [
        {
          id: "api",
          provider: "mcp",
          transport: {
            kind: "shttp",
            url: "https://example.com/mcp",
            apiKeyOptional: true,
          },
          auth: { strategy: "api_key", apiKeyOptional: true },
        },
      ],
    };

    renderWith(<InstallServerModal existingServers={[]} entry={entry} onClose={vi.fn()} />);
    await screen.findByTestId("mcp-install-modal");

    await waitFor(() => expect(SettingsService.getSettings).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    // In tests i18n keys are returned as-is, so the button shows the key name.
    await waitFor(() =>
      expect(screen.getByTestId("mcp-install-submit")).toHaveTextContent(
        "MCP$VERIFYING",
      ),
    );
  });

  // ---------------------------------------------------------------------------
  // Save-as-secret toggle behaviour
  // ---------------------------------------------------------------------------

  // Synthetic stdio entry with one password-type envField, one text-type
  // envField, and one argField. This gives us complete control over field
  // types without depending on the live integration catalog.
  const STDIO_ENTRY = {
    id: "synthetic-stdio",
    name: "Synthetic Stdio Server",
    description: "Stdio server used to test the save-as-secret feature.",
    iconBg: "#000000",
    connectionOptions: [
      {
        id: "stdio",
        provider: "mcp",
        transport: {
          kind: "stdio",
          serverName: "test-server",
          command: "npx",
          args: ["-y", "test-mcp"],
          envFields: [
            {
              key: "API_KEY",
              label: "API Key",
              type: "password",
              required: true,
              placeholder: "Enter API key",
            },
            {
              key: "USERNAME",
              label: "Username",
              type: "text",
              required: false,
              placeholder: "Enter username",
            },
          ],
          argFields: [
            {
              key: "EXTRA_ARG",
              label: "Extra Arg",
              type: "text",
              required: false,
              placeholder: "optional",
            },
          ],
        },
        auth: { strategy: "api_key", apiKeyOptional: true },
      },
    ],
  } as unknown as MarketplaceEntry;

  const SHTTP_ENTRY = {
    id: "synthetic-shttp-secret",
    name: "Synthetic Hosted Server",
    description: "Hosted server used to test credential secret saving.",
    iconBg: "#000000",
    connectionOptions: [
      {
        id: "api",
        provider: "mcp",
        transport: {
          kind: "shttp",
          url: "https://example.com/mcp",
        },
        auth: {
          strategy: "api_key",
          credentialLabel: "Personal access token",
          credentialPlaceholder: "pat_...",
          credentialHelp: "Token from the provider settings.",
          credentialSecretName: "PROVIDER_PERSONAL_ACCESS_TOKEN",
          saveCredentialAsSecretByDefault: true,
        },
      },
    ],
  } as unknown as MarketplaceEntry;

  describe("InstallServerModal — save as secret", () => {
    beforeEach(() => {
      vi.spyOn(SecretsService, "createSecret").mockResolvedValue();
    });

    it("pre-checks the toggle for password-type envFields", async () => {
      renderWith(<InstallServerModal existingServers={[]} entry={STDIO_ENTRY} onClose={vi.fn()} />);
      await screen.findByTestId("mcp-install-modal");

      const toggle = screen.getByTestId("mcp-install-save-secret-API_KEY");
      expect(toggle.querySelector("input[type='checkbox']")).toBeChecked();
    });

    it("leaves non-password envFields unchecked by default", async () => {
      renderWith(<InstallServerModal existingServers={[]} entry={STDIO_ENTRY} onClose={vi.fn()} />);
      await screen.findByTestId("mcp-install-modal");

      const toggle = screen.getByTestId("mcp-install-save-secret-USERNAME");
      expect(toggle.querySelector("input[type='checkbox']")).not.toBeChecked();
    });

    it("does not render a toggle for argFields", async () => {
      renderWith(<InstallServerModal existingServers={[]} entry={STDIO_ENTRY} onClose={vi.fn()} />);
      await screen.findByTestId("mcp-install-modal");

      expect(
        screen.queryByTestId("mcp-install-save-secret-EXTRA_ARG"),
      ).not.toBeInTheDocument();
    });

    it("toggling the checkbox updates its checked state", async () => {
      renderWith(<InstallServerModal existingServers={[]} entry={STDIO_ENTRY} onClose={vi.fn()} />);
      await screen.findByTestId("mcp-install-modal");

      // USERNAME starts unchecked; clicking it should flip to checked.
      const toggle = screen.getByTestId("mcp-install-save-secret-USERNAME");
      const checkbox = toggle.querySelector(
        "input[type='checkbox']",
      ) as HTMLInputElement;
      expect(checkbox).not.toBeChecked();

      fireEvent.click(checkbox);

      expect(checkbox).toBeChecked();
    });

    it("setValue preserves savedAsSecret state when a field value changes", async () => {
      // Before the ...prev bug-fix in setValue, calling onChange on any field
      // would reset savedAsSecret to {}, unchecking all toggles silently.
      renderWith(<InstallServerModal existingServers={[]} entry={STDIO_ENTRY} onClose={vi.fn()} />);
      await screen.findByTestId("mcp-install-modal");

      // API_KEY starts pre-checked. Typing a new value should leave it checked.
      fireEvent.change(screen.getByTestId("mcp-install-field-API_KEY"), {
        target: { value: "new-value" },
      });

      const toggle = screen.getByTestId("mcp-install-save-secret-API_KEY");
      expect(toggle.querySelector("input[type='checkbox']")).toBeChecked();
    });

    it("calls createSecret for checked envFields after a successful install", async () => {
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
      const onClose = vi.fn();
      renderWith(<InstallServerModal existingServers={[]} entry={STDIO_ENTRY} onClose={onClose} />);
      await screen.findByTestId("mcp-install-modal");
      await waitFor(() =>
        expect(SettingsService.getSettings).toHaveBeenCalled(),
      );

      // Fill in the required password field (API_KEY is pre-checked as secret).
      fireEvent.change(screen.getByTestId("mcp-install-field-API_KEY"), {
        target: { value: "my-api-key" },
      });
      fireEvent.click(screen.getByTestId("mcp-install-submit"));

      await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
      await waitFor(() =>
        expect(SecretsService.createSecret).toHaveBeenCalledWith(
          "API_KEY",
          "my-api-key",
          "API Key",
        ),
      );
      // USERNAME was unchecked, so no secret call for it.
      expect(SecretsService.createSecret).not.toHaveBeenCalledWith(
        "USERNAME",
        expect.anything(),
        expect.anything(),
      );
    });

    it("saves hosted MCP credentials as named secrets when configured", async () => {
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
      const onClose = vi.fn();
      renderWith(<InstallServerModal existingServers={[]} entry={SHTTP_ENTRY} onClose={onClose} />);
      await screen.findByTestId("mcp-install-modal");
      await waitFor(() =>
        expect(SettingsService.getSettings).toHaveBeenCalled(),
      );

      expect(screen.getByTestId("mcp-install-field-url")).toHaveValue(
        "https://example.com/mcp",
      );
      expect(screen.getByLabelText("Personal access token")).toHaveAttribute(
        "placeholder",
        "pat_...",
      );
      expect(
        screen.getByText("Token from the provider settings."),
      ).toBeInTheDocument();

      const toggle = screen.getByTestId(
        "mcp-install-save-secret-PROVIDER_PERSONAL_ACCESS_TOKEN",
      );
      expect(toggle.querySelector("input[type='checkbox']")).toBeChecked();

      fireEvent.change(screen.getByTestId("mcp-install-field-api_key"), {
        target: { value: "hosted-token" },
      });
      fireEvent.click(screen.getByTestId("mcp-install-submit"));

      await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
      await waitFor(() =>
        expect(SecretsService.createSecret).toHaveBeenCalledWith(
          "PROVIDER_PERSONAL_ACCESS_TOKEN",
          "hosted-token",
          "Personal access token",
        ),
      );
    });

    it("waits for hosted credential secrets before reporting install success", async () => {
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
      let resolveSecret!: () => void;
      const secretSaved = new Promise<void>((resolve) => {
        resolveSecret = resolve;
      });
      vi.spyOn(SecretsService, "createSecret").mockReturnValue(secretSaved);
      const onClose = vi.fn();
      const onSuccess = vi.fn();

      renderWith(
        <InstallServerModal
          existingServers={[]}
          entry={SHTTP_ENTRY}
          onClose={onClose}
          onSuccess={onSuccess}
        />,
      );
      await screen.findByTestId("mcp-install-modal");
      await waitFor(() =>
        expect(SettingsService.getSettings).toHaveBeenCalled(),
      );

      fireEvent.change(screen.getByTestId("mcp-install-field-api_key"), {
        target: { value: "hosted-token" },
      });
      fireEvent.click(screen.getByTestId("mcp-install-submit"));

      await waitFor(() =>
        expect(SecretsService.createSecret).toHaveBeenCalledWith(
          "PROVIDER_PERSONAL_ACCESS_TOKEN",
          "hosted-token",
          "Personal access token",
        ),
      );
      expect(onSuccess).not.toHaveBeenCalled();
      expect(onClose).not.toHaveBeenCalled();

      resolveSecret();

      await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it("does not call createSecret when all toggles are unchecked before install", async () => {
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
      const onClose = vi.fn();
      renderWith(<InstallServerModal existingServers={[]} entry={STDIO_ENTRY} onClose={onClose} />);
      await screen.findByTestId("mcp-install-modal");
      await waitFor(() =>
        expect(SettingsService.getSettings).toHaveBeenCalled(),
      );

      fireEvent.change(screen.getByTestId("mcp-install-field-API_KEY"), {
        target: { value: "my-api-key" },
      });

      // Uncheck the pre-checked API_KEY toggle before submitting.
      const toggle = screen.getByTestId("mcp-install-save-secret-API_KEY");
      fireEvent.click(toggle.querySelector("input[type='checkbox']")!);

      fireEvent.click(screen.getByTestId("mcp-install-submit"));
      await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));

      // Flush the fire-and-forget microtask chain.
      await Promise.resolve();
      await Promise.resolve();
      expect(SecretsService.createSecret).not.toHaveBeenCalled();
    });

    it("closes the modal even when the secret save fails", async () => {
      vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
      vi.spyOn(SecretsService, "createSecret").mockRejectedValue(
        new Error("forbidden"),
      );
      const onClose = vi.fn();
      renderWith(<InstallServerModal existingServers={[]} entry={STDIO_ENTRY} onClose={onClose} />);
      await screen.findByTestId("mcp-install-modal");
      await waitFor(() =>
        expect(SettingsService.getSettings).toHaveBeenCalled(),
      );

      fireEvent.change(screen.getByTestId("mcp-install-field-API_KEY"), {
        target: { value: "my-api-key" },
      });
      fireEvent.click(screen.getByTestId("mcp-install-submit"));

      // The modal must close regardless of the secret-save outcome.
      await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
      // Secret save errors use toasts, not the modal inline error element.
      expect(
        screen.queryByTestId("mcp-install-modal-error"),
      ).not.toBeInTheDocument();
    });
  });
});
