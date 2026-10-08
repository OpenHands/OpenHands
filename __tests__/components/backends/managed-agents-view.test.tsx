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
});
