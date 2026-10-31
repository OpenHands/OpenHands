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
  it("offers sign-in when an agent-list request discovers a revoked OAuth grant", async () => {
    const connection = { ...CONNECTION, kind: "oauth" as const };
    const bridge = fakeBridge({
      ...SIGNED_IN,
      canUseOAuth: true,
      connections: [connection],
      active: connection,
    });
    vi.mocked(bridge.listAgentConfigs).mockImplementation(async () => {
      const expired = { ...connection, isExpired: true };
      vi.mocked(bridge.getAuthState).mockResolvedValue({
        ...SIGNED_IN,
        canUseOAuth: true,
        connections: [expired],
        active: expired,
      });
      throw new Error("Sign in to DigitalOcean first.");
    });
    window.marsBridge = bridge;

    renderView();

    expect(
      await screen.findByTestId("managed-agents-oauth"),
    ).toBeInTheDocument();
  });

  it.each(["still-valid", "unavailable"] as const)(
    "keeps the original list failure when the auth recheck is %s",
    async (authRecheck) => {
      const bridge = fakeBridge({ ...SIGNED_IN, canUseOAuth: true });
      vi.mocked(bridge.listAgentConfigs).mockImplementation(async () => {
        if (authRecheck === "unavailable") {
          vi.mocked(bridge.getAuthState).mockRejectedValue(
            new Error("Auth state could not be loaded."),
          );
        }
        throw new Error("DigitalOcean is temporarily unreachable.");
      });
      window.marsBridge = bridge;

      renderView();

      await waitFor(() => expect(bridge.getAuthState).toHaveBeenCalledTimes(2));
      expect(
        await screen.findByText("DigitalOcean is temporarily unreachable."),
      ).toBeInTheDocument();
      expect(screen.queryByTestId("managed-agents-oauth")).toBeNull();
    },
  );

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

  describe("signing in with DigitalOcean", () => {
    const OAUTH_SIGNED_OUT: MarsAuthState = {
      ...SIGNED_OUT,
      canUseOAuth: true,
    };
    const AUTHORIZE_URL =
      "https://cloud.digitalocean.com/v1/oauth/authorize?state=s1";

    function fakeTab() {
      return {
        closed: false,
        opener: {} as unknown,
        location: { href: "about:blank" },
        close: vi.fn(),
      };
    }

    /** awaitSignIn stays pending until the test settles it. */
    function oauthBridge({ returnsHere = true } = {}) {
      const bridge = fakeBridge(OAUTH_SIGNED_OUT);
      let settle: (value: unknown) => void = () => {};
      Object.assign(bridge, {
        beginSignIn: vi.fn(async () => ({
          attemptId: "a1",
          authorizeUrl: AUTHORIZE_URL,
          returnsHere,
        })),
        awaitSignIn: vi.fn(
          () =>
            new Promise((resolve) => {
              settle = resolve;
            }),
        ),
        completeSignIn: vi.fn(async () => {
          settle({ status: "done", authState: SIGNED_IN });
          return SIGNED_IN;
        }),
        cancelSignIn: vi.fn(async () => {
          settle({ status: "failed", message: "Sign-in was cancelled." });
        }),
      });
      return {
        bridge,
        finish: () => settle({ status: "done", authState: SIGNED_IN }),
      };
    }

    it("offers DigitalOcean first, with the token behind a disclosure", async () => {
      window.marsBridge = oauthBridge().bridge;
      renderView();

      expect(await screen.findByTestId("managed-agents-oauth")).toBeEnabled();
      expect(screen.queryByTestId("managed-agents-token")).toBeNull();

      await userEvent.click(screen.getByTestId("managed-agents-use-token"));

      expect(screen.getByTestId("managed-agents-token")).toBeInTheDocument();
    });

    it("opens DigitalOcean in the tab the click made, then shows the agents", async () => {
      const { bridge, finish } = oauthBridge();
      window.marsBridge = bridge;
      const tab = fakeTab();
      const open = vi
        .spyOn(window, "open")
        .mockReturnValue(tab as unknown as Window);
      renderView();

      await userEvent.click(await screen.findByTestId("managed-agents-oauth"));

      expect(open).toHaveBeenCalledWith("about:blank", "_blank");
      await waitFor(() => expect(tab.location.href).toBe(AUTHORIZE_URL));
      expect(tab.opener).toBeNull();
      expect(
        screen.getByTestId("managed-agents-pending-sign-in"),
      ).toHaveTextContent("DO_AGENTS$FINISH_ON_DO");
      vi.mocked(bridge.getAuthState).mockResolvedValue(SIGNED_IN);
      finish();
      expect(
        await screen.findByTestId("managed-agents-session-sess_ready"),
      ).toBeInTheDocument();
    });

    it("finishes from a pasted address when the return cannot reach this device", async () => {
      const { bridge } = oauthBridge({ returnsHere: false });
      window.marsBridge = bridge;
      vi.spyOn(window, "open").mockReturnValue(null);
      renderView();
      await userEvent.click(await screen.findByTestId("managed-agents-oauth"));
      expect(
        await screen.findByTestId("managed-agents-pending-sign-in"),
      ).toHaveTextContent("DO_AGENTS$RETURN_WONT_LOAD");
      expect(screen.getByTestId("managed-agents-open-sign-in")).toHaveAttribute(
        "href",
        AUTHORIZE_URL,
      );

      vi.mocked(bridge.getAuthState).mockResolvedValue(SIGNED_IN);
      await userEvent.type(
        screen.getByTestId("managed-agents-return-url"),
        "http://127.0.0.1:3000/mars/oauth/callback?code=c1&state=s1",
      );
      await userEvent.click(screen.getByTestId("managed-agents-return-submit"));

      expect(bridge.completeSignIn).toHaveBeenCalledWith(
        "http://127.0.0.1:3000/mars/oauth/callback?code=c1&state=s1",
        "a1",
      );
      expect(
        await screen.findByTestId("managed-agents-session-sess_ready"),
      ).toBeInTheDocument();
    });

    it("cancels quietly", async () => {
      const { bridge } = oauthBridge();
      window.marsBridge = bridge;
      vi.spyOn(window, "open").mockReturnValue(null);
      renderView();
      await userEvent.click(await screen.findByTestId("managed-agents-oauth"));
      await screen.findByTestId("managed-agents-pending-sign-in");

      await userEvent.click(screen.getByTestId("managed-agents-sign-in-abort"));

      expect(bridge.cancelSignIn).toHaveBeenCalledWith("a1");
      await waitFor(() =>
        expect(
          screen.queryByTestId("managed-agents-pending-sign-in"),
        ).toBeNull(),
      );
      expect(screen.queryByTestId("managed-agents-auth-error")).toBeNull();
      expect(screen.getByTestId("managed-agents-oauth")).toBeEnabled();
    });

    it("stays on Add connection after a sign-in started there is cancelled", async () => {
      const { bridge } = oauthBridge();
      vi.mocked(bridge.getAuthState).mockResolvedValue({
        ...SIGNED_IN,
        canUseOAuth: true,
      });
      window.marsBridge = bridge;
      vi.spyOn(window, "open").mockReturnValue(null);
      renderView();
      await userEvent.click(
        await screen.findByTestId("managed-agents-add-connection"),
      );
      await userEvent.click(screen.getByTestId("managed-agents-oauth"));
      await screen.findByTestId("managed-agents-pending-sign-in");

      await userEvent.click(screen.getByTestId("managed-agents-sign-in-abort"));

      await waitFor(() =>
        expect(
          screen.queryByTestId("managed-agents-pending-sign-in"),
        ).toBeNull(),
      );
      expect(screen.getByTestId("managed-agents-sign-in")).toBeInTheDocument();
      expect(
        screen.getByTestId("managed-agents-sign-in-cancel"),
      ).toBeInTheDocument();
    });

    it("takes a token while a browser sign-in waits, and drops that sign-in", async () => {
      const { bridge } = oauthBridge();
      window.marsBridge = bridge;
      vi.spyOn(window, "open").mockReturnValue(null);
      renderView();
      await userEvent.click(await screen.findByTestId("managed-agents-oauth"));
      await screen.findByTestId("managed-agents-pending-sign-in");
      await userEvent.click(screen.getByTestId("managed-agents-use-token"));
      await userEvent.type(
        screen.getByTestId("managed-agents-token"),
        `dop_v1_${"a".repeat(64)}`,
      );

      vi.mocked(bridge.getAuthState).mockResolvedValue(SIGNED_IN);
      await userEvent.click(screen.getByTestId("managed-agents-token-submit"));

      expect(bridge.cancelSignIn).toHaveBeenCalledWith("a1");
      expect(bridge.savePat).toHaveBeenCalledWith({
        token: `dop_v1_${"a".repeat(64)}`,
      });
      expect(
        await screen.findByTestId("managed-agents-session-sess_ready"),
      ).toBeInTheDocument();
    });
  });
});
