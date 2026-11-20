import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
} from "@testing-library/react";
import { AxiosError } from "axios";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import McpService from "#/api/mcp-service/mcp-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  __resetMcpHealthStoreForTests,
  getMcpHealthSnapshot,
  setMcpServerHealth,
} from "#/api/mcp-health/mcp-health-store";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { CustomServerEditor } from "#/components/features/mcp-page/custom-server-editor";
import { useSettings } from "#/hooks/query/use-settings";
import { createAgentServerQueryClient } from "#/query-client-config";
import * as toastHandlers from "#/utils/custom-toast-handlers";

import type { Settings } from "#/types/settings";
import type { MCPServerConfig } from "#/types/mcp-server";

const EDIT_STDIO_SERVER: MCPServerConfig = {
  id: "github",
  type: "stdio",
  name: "github",
  command: "docker",
  args: ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"],
};

const EDIT_OAUTH_SERVER: MCPServerConfig = {
  id: "superhuman_mail",
  type: "shttp",
  name: "superhuman_mail",
  url: "https://mcp.mail.superhuman.com/mcp",
  auth: {
    strategy: "oauth2",
    authentication: { type: "oauth", client_auth_method: "none" },
  },
};

const EDIT_REMOTE_SERVER: MCPServerConfig = {
  id: "jira",
  type: "shttp",
  name: "jira",
  url: "https://mcp-jira.example.com/mcp",
  auth: { strategy: "bearer", value: "**********" },
};

function buildSettingsWithMcp(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    agent_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
      mcp_config: {
        github: {
          command: "docker",
          args: ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"],
        },
        jira: {
          url: "https://mcp-jira.example.com/mcp",
          transport: "http",
          auth: { strategy: "bearer", value: "**********" },
        },
        superhuman_mail: {
          url: "https://mcp.mail.superhuman.com/mcp",
          transport: "http",
          auth: {
            strategy: "oauth2",
            authentication: { type: "oauth", client_auth_method: "none" },
          },
        },
      },
    },
    ...overrides,
  };
}

/**
 * Wrapper that only mounts the editor once `useSettings` has resolved.
 * `useAddMcpServer`'s `mutationFn` silently no-ops when settings is
 * undefined (and that no-op resolves, triggering the per-call
 * `onSuccess` → which would close our modal). Waiting for the query
 * makes the test deterministic.
 */
function EditorOnceSettingsLoaded({ onClose }: { onClose: () => void }) {
  const { data } = useSettings();
  if (!data) return null;
  return (
    <CustomServerEditor
      server={{ id: "", type: "sse" }}
      existingServers={[]}
      onClose={onClose}
    />
  );
}

function EditEditorOnceSettingsLoaded({ onClose }: { onClose: () => void }) {
  const { data } = useSettings();
  if (!data) return null;
  return (
    <CustomServerEditor
      server={EDIT_STDIO_SERVER}
      existingServers={[EDIT_STDIO_SERVER]}
      onClose={onClose}
    />
  );
}

function EditRemoteEditorOnceSettingsLoaded({
  onClose,
}: {
  onClose: () => void;
}) {
  const { data } = useSettings();
  if (!data) return null;
  return (
    <CustomServerEditor
      server={EDIT_REMOTE_SERVER}
      existingServers={[EDIT_REMOTE_SERVER]}
      onClose={onClose}
    />
  );
}

function EditOAuthEditorOnceSettingsLoaded({
  onClose,
}: {
  onClose: () => void;
}) {
  const { data } = useSettings();
  if (!data) return null;
  return (
    <CustomServerEditor
      server={EDIT_OAUTH_SERVER}
      existingServers={[EDIT_OAUTH_SERVER]}
      onClose={onClose}
    />
  );
}

function renderWith(
  ui: React.ReactNode,
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) {
  return render(ui, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={client}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });
}

describe("CustomServerEditor", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    // The backend registry persists to localStorage; wipe it before the
    // reset re-reads storage so each test starts on the default local backend.
    window.localStorage.clear();
    __resetActiveStoreForTests();
    vi.spyOn(SettingsService, "createMcpServer").mockImplementation(
      (settingsKey, server) =>
        SettingsService.saveSettings({
          agent_settings_diff: {
            mcp_config: { [settingsKey]: server },
          },
        }),
    );
    vi.spyOn(SettingsService, "patchMcpServer").mockImplementation(
      (settingsKey, patch) =>
        SettingsService.saveSettings({
          agent_settings_diff: {
            mcp_config: { [settingsKey]: patch },
          },
        }),
    );
    vi.spyOn(SettingsService, "deleteMcpServer").mockImplementation(
      (settingsKey) =>
        SettingsService.saveSettings({
          agent_settings_diff: {
            mcp_config: { [settingsKey]: null },
          },
        }),
    );
    vi.spyOn(SettingsService, "patchMcpConfig").mockImplementation((patch) =>
      SettingsService.saveSettings({
        agent_settings_diff: { mcp_config: patch },
      }),
    );
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
    // Pre-flight connectivity test must pass so the save mutation is reached.
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: true,
      tools: [],
    });
  });

  it("shows one error toast when Save cannot test the connection", async () => {
    // Arrange: use the app's global mutation handler and the real editor hooks.
    vi.spyOn(McpService, "testServer").mockRejectedValue(
      new AxiosError("Agent Server unavailable"),
    );
    const errorToast = vi
      .spyOn(toastHandlers, "displayErrorToast")
      .mockImplementation(() => {});
    const onClose = vi.fn();
    renderWith(
      <EditEditorOnceSettingsLoaded onClose={onClose} />,
      createAgentServerQueryClient(),
    );
    await screen.findByTestId("mcp-custom-editor");

    // Act
    fireEvent.click(screen.getByTestId("submit-button"));

    // Assert: the failure is reported once and no save or dismissal follows.
    await waitFor(() => expect(errorToast).toHaveBeenCalled());
    await waitFor(() =>
      expect(screen.getByTestId("submit-button")).toBeEnabled(),
    );
    expect(errorToast.mock.calls).toEqual([["Agent Server unavailable"]]);
    expect(SettingsService.createMcpServer).not.toHaveBeenCalled();
    expect(SettingsService.patchMcpServer).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByTestId("mcp-custom-editor")).toBeInTheDocument();
  });

  it("has a single Save action that tests and saves the server", async () => {
    // Arrange: a remote server whose connection test passes.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    const testSpy = vi
      .spyOn(McpService, "testServer")
      .mockResolvedValue({ ok: true, tools: ["search_issues"] });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const onClose = vi.fn();
    renderWith(<EditRemoteEditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");

    // Act
    fireEvent.click(screen.getByTestId("submit-button"));

    // Assert: one click both tests and persists, then closes the editor.
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(testSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId("mcp-test-connection")).not.toBeInTheDocument();
  });

  it("keeps the modal open and does not call onClose when the add mutation fails", async () => {
    // Simulate a backend rejection — the editor should surface the
    // failure as an `onError` toast and leave the modal open so the
    // user can retry. Previously these calls had no `onError` at
    // all, and the modal closed even on a 4xx/5xx because
    // tanstack-query's per-call `onSuccess` doesn't run on
    // rejection but didn't gate the close either way.
    const err = new AxiosError("Boom");
    err.response = {
      status: 400,
      data: { detail: "Server name already in use" },
    } as unknown as AxiosError["response"];
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockRejectedValue(err);

    const onClose = vi.fn();
    renderWith(<EditorOnceSettingsLoaded onClose={onClose} />);

    // Wrapper waits for useSettings before mounting the editor, so
    // by the time we see the editor the mutation hook will fire its
    // mutationFn (rather than silently no-op).
    await screen.findByTestId("mcp-custom-editor");
    fireEvent.change(screen.getByTestId("url-input"), {
      target: { value: "https://example.com/mcp" },
    });
    fireEvent.click(screen.getByTestId("submit-button"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));

    // Modal is still mounted — onClose was *not* called on failure.
    await waitFor(() => {
      expect(screen.queryByTestId("mcp-custom-editor")).toBeInTheDocument();
      expect(onClose).not.toHaveBeenCalled();
    });
  });

  it("closes from the top-right close button", async () => {
    const onClose = vi.fn();
    renderWith(<EditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");

    fireEvent.click(screen.getByTestId("mcp-custom-editor-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not show delete in add mode", async () => {
    renderWith(<EditorOnceSettingsLoaded onClose={vi.fn()} />);
    await screen.findByTestId("mcp-custom-editor");

    expect(
      screen.queryByTestId("mcp-custom-editor-delete"),
    ).not.toBeInTheDocument();
  });

  it("deletes an installed server after confirmation", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    const onClose = vi.fn();
    renderWith(<EditEditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");

    fireEvent.click(screen.getByTestId("mcp-custom-editor-delete"));
    expect(await screen.findByTestId("confirmation-modal")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("confirm-button"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  it("calls onClose when the header close button is clicked", async () => {
    const onClose = vi.fn();
    renderWith(<EditorOnceSettingsLoaded onClose={onClose} />);

    fireEvent.click(await screen.findByTestId("mcp-custom-editor-close"));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("surfaces a credential failure from Save in the edit modal without saving", async () => {
    // Arrange: editing an installed server whose stored credentials the
    // verification call rejects — previously this path always reported
    // "Connected" because only tools/list was exercised.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: false,
      error: "invalid_auth",
      error_kind: "credentials",
    });

    const onClose = vi.fn();
    renderWith(<EditEditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");

    // Act: save, which tests the connection first.
    fireEvent.click(screen.getByTestId("submit-button"));

    // Assert: the credentials-specific message is shown inline (i18n keys
    // are returned as-is in tests) and nothing is saved.
    await waitFor(() =>
      expect(screen.getByTestId("mcp-test-message")).toHaveTextContent(
        "MCP$TEST_ERROR_CREDENTIALS",
      ),
    );
    expect(SettingsService.patchMcpServer).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("explains a stdio connection failure from Save as a command that could not start", async () => {
    // Arrange: editing an installed stdio server whose command cannot spawn.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: false,
      error: "Client failed to connect: [Errno 2] No such file or directory",
      error_kind: "connection",
    });
    renderWith(<EditEditorOnceSettingsLoaded onClose={vi.fn()} />);
    await screen.findByTestId("mcp-custom-editor");

    // Act
    fireEvent.click(screen.getByTestId("submit-button"));

    // Assert: the stdio-specific message, not the URL hint.
    await waitFor(() =>
      expect(screen.getByTestId("mcp-test-message")).toHaveTextContent(
        "MCP$TEST_ERROR_STDIO_START",
      ),
    );
  });

  it("tests remote servers through the app server before saving on cloud backends", async () => {
    // Arrange: a cloud backend is active and the probe (routed through the
    // app server) succeeds.
    setRegisteredBackends([
      {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "k",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud-1" });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    const testSpy = vi
      .spyOn(McpService, "testServer")
      .mockResolvedValue({ ok: true, tools: ["search_issues"] });
    const onClose = vi.fn();
    renderWith(<EditRemoteEditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");

    // Act
    fireEvent.click(screen.getByTestId("submit-button"));

    // Assert
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(testSpy).toHaveBeenCalledWith(
      expect.objectContaining({ type: "shttp", url: EDIT_REMOTE_SERVER.url }),
    );
  });

  it("saves stdio servers on cloud backends without reporting a synthetic health verdict", async () => {
    setRegisteredBackends([
      {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "k",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud-1" });
    __resetMcpHealthStoreForTests();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    const onClose = vi.fn();
    renderWith(<EditEditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");

    // Act
    fireEvent.click(screen.getByTestId("submit-button"));

    // Assert: saved and closed, but the card is not marked healthy.
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(
      getMcpHealthSnapshot()[getMcpServerHealthKey(EDIT_STDIO_SERVER)],
    ).toBeUndefined();
  });

  it("reseeds the edited server's health from the fresh pre-save test", async () => {
    // Arrange: the installed card shows a failure; the user re-saves the
    // server (e.g. after fixing the credential) and the pre-save test now
    // passes. The card must flip to healthy without a page reload — the
    // server's own prior entry is overwritten, not duplicate-guarded.
    __resetMcpHealthStoreForTests();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    const key = getMcpServerHealthKey(EDIT_STDIO_SERVER);
    setMcpServerHealth(key, {
      status: "failed",
      kind: "credentials",
      error: "invalid_auth",
      checkedAt: 1,
    });

    const onClose = vi.fn();
    renderWith(<EditEditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");

    // Act: save without structural changes (test passes via the beforeEach
    // McpService.testServer mock).
    fireEvent.click(screen.getByTestId("submit-button"));

    // Assert: the same health key now carries the fresh healthy verdict.
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(getMcpHealthSnapshot()[key]).toMatchObject({ status: "healthy" });
  });

  it("runs the OAuth flow from Save on cloud backends", async () => {
    // Arrange: a cloud backend is active and an OAuth server is being edited.
    setRegisteredBackends([
      {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "k",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud-1" });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    const authorizeSpy = vi
      .spyOn(McpService, "authorizeOAuth")
      .mockResolvedValue({ ok: true, tools: ["search_mail"] });
    const testSpy = vi
      .spyOn(McpService, "testServer")
      .mockResolvedValue({ ok: true, tools: [] });
    vi.spyOn(SettingsService, "saveSettings").mockResolvedValue(true);
    const onClose = vi.fn();
    renderWith(<EditOAuthEditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");

    // Act
    fireEvent.click(screen.getByTestId("submit-button"));

    // Assert
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(authorizeSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "shttp",
        url: EDIT_OAUTH_SERVER.url,
        auth: expect.objectContaining({ strategy: "oauth2" }),
      }),
    );
    expect(testSpy).not.toHaveBeenCalled();
  });

  it("persists OAuth state returned by the connection test when editing", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          mcp_config: {
            superhuman_mail: {
              url: "https://mcp.mail.superhuman.com/mcp",
              transport: "http",
              auth: {
                strategy: "oauth2",
                authentication: { type: "oauth", client_auth_method: "none" },
              },
            },
          },
        },
      }),
    );
    vi.spyOn(McpService, "authorizeOAuth").mockResolvedValue({
      ok: true,
      tools: [],
      oauth_state: {
        tokens: { access_token: "gAAAAencrypted-access-token" },
        token_expires_at: 12345,
      },
    });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderWith(<EditOAuthEditorOnceSettingsLoaded onClose={vi.fn()} />);
    await screen.findByTestId("mcp-custom-editor");
    fireEvent.click(screen.getByTestId("submit-button"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    const sent = (saveSpy.mock.calls[0][0] as Record<string, unknown>)
      .agent_settings_diff as {
      mcp_config: Record<string, unknown>;
    };
    expect(sent.mcp_config).toMatchObject({
      superhuman_mail: {
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
  it("offers Save anyway after a failed test and saves the tested server", async () => {
    // Arrange: a server whose connection test fails, e.g. one only reachable
    // from inside the sandbox.
    __resetMcpHealthStoreForTests();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: false,
      error: "connect ECONNREFUSED",
      error_kind: "connection",
    });
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const onClose = vi.fn();
    renderWith(<EditRemoteEditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");
    fireEvent.click(screen.getByTestId("submit-button"));
    await screen.findByTestId("mcp-save-anyway");
    expect(saveSpy).not.toHaveBeenCalled();

    // Act
    fireEvent.click(screen.getByTestId("mcp-save-anyway"));

    // Assert: saved without another test, and the card shows the failure.
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(McpService.testServer).toHaveBeenCalledTimes(1);
    expect(
      getMcpHealthSnapshot()[getMcpServerHealthKey(EDIT_REMOTE_SERVER)],
    ).toMatchObject({ status: "failed" });
  });

  it("does not offer Save anyway when OAuth sign-in fails", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    vi.spyOn(McpService, "authorizeOAuth").mockResolvedValue({
      ok: false,
      error: "popup closed",
      error_kind: "authorization",
    });
    renderWith(<EditOAuthEditorOnceSettingsLoaded onClose={vi.fn()} />);
    await screen.findByTestId("mcp-custom-editor");

    fireEvent.click(screen.getByTestId("submit-button"));

    await screen.findByTestId("mcp-test-message");
    expect(screen.queryByTestId("mcp-save-anyway")).not.toBeInTheDocument();
  });

  it("clears a failed test result once the user edits the form", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettingsWithMcp(),
    );
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: false,
      error: "connect ECONNREFUSED",
      error_kind: "connection",
    });
    renderWith(<EditRemoteEditorOnceSettingsLoaded onClose={vi.fn()} />);
    await screen.findByTestId("mcp-custom-editor");
    fireEvent.click(screen.getByTestId("submit-button"));
    await screen.findByTestId("mcp-save-anyway");

    fireEvent.change(screen.getByTestId("url-input"), {
      target: { value: "https://mcp-jira.example.com/v2/mcp" },
    });

    expect(screen.queryByTestId("mcp-test-message")).not.toBeInTheDocument();
    expect(screen.queryByTestId("mcp-save-anyway")).not.toBeInTheDocument();
  });

  it("asks before discarding unsaved edits when the editor is closed", async () => {
    // Arrange: the user has typed into the form but not saved.
    const onClose = vi.fn();
    renderWith(<EditorOnceSettingsLoaded onClose={onClose} />);
    await screen.findByTestId("mcp-custom-editor");
    fireEvent.change(screen.getByTestId("url-input"), {
      target: { value: "https://example.com/mcp" },
    });

    // Act: close from the top-right button, then keep editing.
    fireEvent.click(screen.getByTestId("mcp-custom-editor-close"));
    expect(await screen.findByTestId("confirmation-modal")).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.click(
      within(screen.getByTestId("confirmation-modal")).getByTestId(
        "cancel-button",
      ),
    );
    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();

    // Act: close again and discard.
    fireEvent.click(screen.getByTestId("mcp-custom-editor-close"));
    fireEvent.click(await screen.findByTestId("confirm-button"));

    // Assert
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
