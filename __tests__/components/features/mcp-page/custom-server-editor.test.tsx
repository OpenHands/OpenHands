import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AxiosError } from "axios";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import McpService from "#/api/mcp-service/mcp-service.api";
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

import type { Settings } from "#/types/settings";
import type { MCPServerConfig } from "#/types/mcp-server";

const EDIT_STDIO_SERVER: MCPServerConfig = {
  id: "stdio-0",
  type: "stdio",
  name: "github",
  command: "docker",
  args: ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"],
};

const EDIT_OAUTH_SERVER: MCPServerConfig = {
  id: "shttp-0",
  type: "shttp",
  name: "superhuman_mail",
  url: "https://mcp.mail.superhuman.com/mcp",
  auth: {
    strategy: "oauth2",
    authentication: { type: "oauth", client_auth_method: "none" },
  },
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

describe("CustomServerEditor", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
    // Pre-flight connectivity test must pass so the save mutation is reached.
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: true,
      tools: [],
    });
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

  it("surfaces a credential failure from Test connection in the edit modal", async () => {
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

    renderWith(<EditEditorOnceSettingsLoaded onClose={vi.fn()} />);
    await screen.findByTestId("mcp-custom-editor");

    // Act: run the connection test from the edit form.
    fireEvent.click(screen.getByTestId("mcp-test-connection"));

    // Assert: the credentials-specific message is shown inline (i18n keys
    // are returned as-is in tests).
    await waitFor(() =>
      expect(screen.getByTestId("mcp-test-message")).toHaveTextContent(
        "MCP$TEST_ERROR_CREDENTIALS",
      ),
    );
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
});
