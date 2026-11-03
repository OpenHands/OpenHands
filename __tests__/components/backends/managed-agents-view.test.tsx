import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";

vi.mock("#/services/telemetry", () => ({
  setTelemetryCloudContext: vi.fn(),
  setTelemetryIdentity: vi.fn(),
}));

const validateLocalBackend = vi.hoisted(() =>
  vi.fn(async (): Promise<string | null> => "1.44.1"),
);
vi.mock("#/api/agent-server-compatibility", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/api/agent-server-compatibility")
  >()),
  validateLocalBackend,
}));

const searchConversations = vi.hoisted(() =>
  vi.fn(async (): Promise<{ items: { id: string }[] }> => ({ items: [] })),
);
vi.mock("@openhands/typescript-client/clients", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@openhands/typescript-client/clients")
  >()),
  ConversationClient: vi.fn(function ConversationClient() {
    return { searchConversations };
  }),
}));

import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { __resetHealthStoreForTests } from "#/api/backend-registry/health-store";
import { BACKENDS_STORAGE_KEY } from "#/api/backend-registry/storage";
import type { Backend } from "#/api/backend-registry/types";
import type {
  MarsAuthState,
  MarsBridge,
  MarsSession,
} from "#/api/mars/mars-tunnel-backend";
import { ManagedAgentsView } from "#/components/features/backends/managed-agents/managed-agents-view";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

const SIGNED_OUT: MarsAuthState = {
  connections: [],
  active: null,
  isPersistent: true,
  canUseOAuth: false,
};

const CONNECTION = {
  id: "conn_1",
  kind: "pat" as const,
  label: "My team",
  teamName: null,
  expiresAt: null,
  isExpired: false,
};

const SIGNED_IN: MarsAuthState = {
  ...SIGNED_OUT,
  connections: [CONNECTION],
  active: CONNECTION,
};

const READY_SESSION: MarsSession = {
  session_id: "sess_ready",
  name: "fix-login",
  status: "SESSION_STATUS_READY",
  config_id: "cfg_1",
};

function fakeBridge(authState: MarsAuthState) {
  return {
    getAuthState: vi.fn(async () => authState),
    savePat: vi.fn(async () => SIGNED_IN),
    listAgentConfigs: vi.fn(async () => ({
      configs: [{ id: "cfg_1", name: "Web agent", agent: "openhands" }],
      nextPageToken: null,
    })),
    listSessions: vi.fn(async () => ({
      sessions: [READY_SESSION],
      nextPageToken: null,
    })),
    listConfigSessions: vi.fn(async () => ({
      sessions: [READY_SESSION],
      nextPageToken: null,
    })),
    openTunnel: vi.fn(async ({ sessionId }: { sessionId: string }) => ({
      sessionId,
      status: "connected" as const,
      remotePort: 8000,
      localPort: 51000,
      error: undefined,
    })),
    closeTunnel: vi.fn(async () => {}),
    getTunnel: vi.fn(async () => undefined),
    destroySession: vi.fn(async () => {}),
    deleteAgentConfig: vi.fn(async () => {}),
    signOut: vi.fn(async () => SIGNED_OUT),
  } as unknown as MarsBridge;
}

function renderView() {
  const navigate = vi.fn();
  const onDone = vi.fn();
  renderWithProviders(
    <ActiveBackendProvider>
      <ManagedAgentsView onBack={vi.fn()} onDone={onDone} />
    </ActiveBackendProvider>,
    { navigation: { navigate } },
  );
  return { navigate, onDone };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
  validateLocalBackend.mockResolvedValue("1.44.1");
  searchConversations.mockResolvedValue({ items: [] });
});

afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
  delete window.marsBridge;
});

describe("ManagedAgentsView", () => {
  it("asks a signed-out user for a token and shows their agents once it is accepted", async () => {
    const bridge = fakeBridge(SIGNED_OUT);
    window.marsBridge = bridge;
    renderView();

    await userEvent.type(
      await screen.findByTestId("managed-agents-token"),
      "dop_v1_token",
    );
    vi.mocked(bridge.getAuthState).mockResolvedValue(SIGNED_IN);
    await userEvent.click(screen.getByTestId("managed-agents-token-submit"));

    expect(bridge.savePat).toHaveBeenCalledWith({ token: "dop_v1_token" });
    expect(
      await screen.findByTestId("managed-agents-session-sess_ready"),
    ).toBeInTheDocument();
  });

  it("lands in the session's latest conversation after connecting", async () => {
    window.marsBridge = fakeBridge(SIGNED_IN);
    searchConversations.mockResolvedValue({ items: [{ id: "conv_42" }] });
    const { navigate, onDone } = renderView();

    await userEvent.click(
      await screen.findByTestId("managed-agents-connect-sess_ready"),
    );

    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(navigate).toHaveBeenCalledWith(
      expect.stringMatching(/^\/conversations\/conv_42\?/),
    );
  });

  it("keeps the agent of a session listed under Other sessions when connecting", async () => {
    // Arrange: the session is missing from its config's own listing.
    const bridge = fakeBridge(SIGNED_IN);
    vi.mocked(bridge.listConfigSessions).mockResolvedValue({
      sessions: [],
      nextPageToken: null,
    });
    window.marsBridge = bridge;
    const { onDone } = renderView();
    expect(
      await screen.findByTestId("managed-agents-other-sessions"),
    ).toBeInTheDocument();

    // Act
    await userEvent.click(
      screen.getByTestId("managed-agents-connect-sess_ready"),
    );

    // Assert
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    const stored: Backend[] = JSON.parse(
      window.localStorage.getItem(BACKENDS_STORAGE_KEY) ?? "[]",
    );
    // Named after the session alone, so the backend menu shows just that.
    expect(
      stored.find((backend) => backend.marsSessionId === "sess_ready"),
    ).toMatchObject({ marsConfigId: "cfg_1", name: "fix-login" });
  });

  it("signs out through the bridge and returns to the sign-in form", async () => {
    const bridge = fakeBridge(SIGNED_IN);
    window.marsBridge = bridge;
    renderView();

    vi.mocked(bridge.getAuthState).mockResolvedValue(SIGNED_OUT);
    await userEvent.click(await screen.findByTestId("managed-agents-sign-out"));

    expect(bridge.signOut).toHaveBeenCalledOnce();
    expect(
      await screen.findByTestId("managed-agents-sign-in"),
    ).toBeInTheDocument();
  });

  it("blocks other connects and launches while a new session is being created", async () => {
    // Arrange
    const bridge = fakeBridge(SIGNED_IN);
    bridge.createSession = vi.fn(() => new Promise<MarsSession>(() => {}));
    window.marsBridge = bridge;
    renderView();
    const newSession = await screen.findByTestId(
      "managed-agents-new-session-cfg_1",
    );

    // Act
    await userEvent.click(newSession);

    // Assert
    expect(bridge.createSession).toHaveBeenCalledOnce();
    expect(newSession).toBeDisabled();
    expect(
      screen.getByTestId("managed-agents-connect-sess_ready"),
    ).toBeDisabled();
  });

  it("explains a refused tunnel on the session row instead of timing out", async () => {
    const bridge = fakeBridge(SIGNED_IN);
    vi.mocked(bridge.getTunnel).mockResolvedValue({
      sessionId: "sess_ready",
      status: "connected",
      remotePort: 8000,
      localPort: 51000,
      error: undefined,
      upstreamFailure: { closeCode: 4001, httpStatus: null, message: "gone" },
    });
    window.marsBridge = bridge;
    validateLocalBackend.mockRejectedValue(new Error("connection reset"));
    const { navigate } = renderView();

    await userEvent.click(
      await screen.findByTestId("managed-agents-connect-sess_ready"),
    );

    expect(
      await screen.findByTestId("managed-agents-session-error-sess_ready"),
    ).toHaveTextContent("DO_AGENTS$ERROR_REFUSED");
    expect(navigate).not.toHaveBeenCalled();
  });

  it("reports a failed connect after launching on the new session's row, not the agent", async () => {
    // Arrange
    const NEW_SESSION: MarsSession = {
      session_id: "sess_new",
      name: "new-session",
      status: "SESSION_STATUS_READY",
      config_id: "cfg_1",
    };
    const bridge = fakeBridge(SIGNED_IN);
    bridge.createSession = vi.fn(async () => NEW_SESSION);
    const sessions = { sessions: [READY_SESSION], nextPageToken: null };
    vi.mocked(bridge.listSessions).mockImplementation(async () => sessions);
    vi.mocked(bridge.listConfigSessions).mockImplementation(
      async () => sessions,
    );
    vi.mocked(bridge.getTunnel).mockResolvedValue({
      sessionId: "sess_new",
      status: "connected",
      remotePort: 8000,
      localPort: 51000,
      error: undefined,
      upstreamFailure: { closeCode: 4001, httpStatus: null, message: "gone" },
    });
    window.marsBridge = bridge;
    validateLocalBackend.mockRejectedValue(new Error("connection reset"));
    const { navigate } = renderView();
    const newSession = await screen.findByTestId(
      "managed-agents-new-session-cfg_1",
    );
    sessions.sessions = [READY_SESSION, NEW_SESSION];

    // Act
    await userEvent.click(newSession);

    // Assert
    expect(
      await screen.findByTestId("managed-agents-session-error-sess_new"),
    ).toHaveTextContent("DO_AGENTS$ERROR_REFUSED");
    // Not also on the agent header, where a later connect would not clear it.
    expect(screen.getAllByText("DO_AGENTS$ERROR_REFUSED")).toHaveLength(1);
    expect(navigate).not.toHaveBeenCalled();
  });

  it("stops a session only after the user confirms, then refreshes the list", async () => {
    const bridge = fakeBridge(SIGNED_IN);
    window.marsBridge = bridge;
    renderView();

    await userEvent.click(
      await screen.findByTestId("managed-agents-stop-sess_ready"),
    );
    // Nothing destructive happens before the confirmation.
    expect(bridge.destroySession).not.toHaveBeenCalled();
    expect(screen.getByTestId("confirmation-modal")).toHaveTextContent(
      "DO_AGENTS$STOP_CONFIRM",
    );

    await userEvent.click(screen.getByTestId("cancel-button"));
    expect(bridge.destroySession).not.toHaveBeenCalled();
    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId("managed-agents-stop-sess_ready"));
    await userEvent.click(screen.getByTestId("confirm-button"));

    await waitFor(() =>
      expect(bridge.destroySession).toHaveBeenCalledWith("sess_ready"),
    );
    await waitFor(() =>
      expect(
        screen.queryByTestId("confirmation-modal"),
      ).not.toBeInTheDocument(),
    );
    // The list is re-read so the stopped session disappears.
    expect(
      vi.mocked(bridge.listConfigSessions).mock.calls.length,
    ).toBeGreaterThan(1);
  });

  it("offers Delete agent only once the agent has no live session", async () => {
    const bridge = fakeBridge(SIGNED_IN);
    window.marsBridge = bridge;
    renderView();

    const deleteButton = await screen.findByTestId(
      "managed-agents-delete-agent-cfg_1",
    );
    expect(deleteButton).toBeDisabled();
    expect(deleteButton).toHaveAttribute(
      "title",
      "DO_AGENTS$DELETE_AGENT_BLOCKED",
    );
    expect(bridge.deleteAgentConfig).not.toHaveBeenCalled();
  });

  it("deletes an agent with no live sessions after confirmation", async () => {
    const bridge = fakeBridge(SIGNED_IN);
    vi.mocked(bridge.listConfigSessions).mockResolvedValue({
      sessions: [{ ...READY_SESSION, status: "SESSION_STATUS_DESTROYED" }],
      nextPageToken: null,
    });
    vi.mocked(bridge.listSessions).mockResolvedValue({
      sessions: [],
      nextPageToken: null,
    });
    window.marsBridge = bridge;
    renderView();

    const deleteButton = await screen.findByTestId(
      "managed-agents-delete-agent-cfg_1",
    );
    await waitFor(() => expect(deleteButton).toBeEnabled());
    await userEvent.click(deleteButton);
    expect(screen.getByTestId("confirmation-modal")).toHaveTextContent(
      "DO_AGENTS$DELETE_AGENT_CONFIRM",
    );
    await userEvent.click(screen.getByTestId("confirm-button"));

    await waitFor(() =>
      expect(bridge.deleteAgentConfig).toHaveBeenCalledWith("cfg_1"),
    );
  });

  it("names the signed-in team and says where its sign-in is kept", async () => {
    const connection = {
      ...CONNECTION,
      teamName: "Research",
      email: "me@example.com",
    };
    window.marsBridge = fakeBridge({
      ...SIGNED_IN,
      connections: [connection],
      active: connection,
      storage: { kind: "keychain", label: "macOS Keychain", host: "studio" },
    });
    renderView();

    const account = await screen.findByTestId("managed-agents-account");
    expect(account).toHaveTextContent("Research · me@example.com");
    expect(
      screen.getByTestId("digitalocean-sign-in-storage"),
    ).toHaveTextContent("DO_AGENTS$SAVED_IN_KEYCHAIN");
  });

  it("warns when the host could not keep the sign-in", async () => {
    window.marsBridge = fakeBridge({
      ...SIGNED_IN,
      active: { ...CONNECTION, isSaved: false },
      storage: { kind: "keychain", label: "macOS Keychain", host: "studio" },
    });
    renderView();

    expect(
      await screen.findByTestId("digitalocean-sign-in-storage"),
    ).toHaveTextContent("DO_AGENTS$NOT_SAVED");
  });

  it("warns when nothing on the host keeps sign-ins", async () => {
    window.marsBridge = fakeBridge({
      ...SIGNED_IN,
      isPersistent: false,
      storage: { kind: "memory", label: "memory", host: "studio" },
    });
    renderView();

    expect(
      await screen.findByTestId("digitalocean-sign-in-storage"),
    ).toHaveTextContent("DO_AGENTS$NOT_SAVED");
  });
});
