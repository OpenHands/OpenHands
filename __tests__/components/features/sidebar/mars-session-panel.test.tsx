import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";

vi.mock("#/services/telemetry", () => ({
  setTelemetryCloudContext: vi.fn(),
  setTelemetryIdentity: vi.fn(),
}));

vi.mock("#/api/agent-server-compatibility", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/api/agent-server-compatibility")
  >()),
  validateLocalBackend: vi.fn(async () => "1.44.1"),
}));

vi.mock("@openhands/typescript-client/clients", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@openhands/typescript-client/clients")
  >()),
  ConversationClient: vi.fn(function ConversationClient() {
    return { searchConversations: vi.fn(async () => ({ items: [] })) };
  }),
}));

import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { __resetHealthStoreForTests } from "#/api/backend-registry/health-store";
import type { Backend } from "#/api/backend-registry/types";
import type { MarsBridge, MarsSession } from "#/api/mars/mars-tunnel-backend";
import { MarsSessionPanel } from "#/components/features/sidebar/mars-session-panel";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";

const CONNECTION = {
  id: "conn_1",
  kind: "pat" as const,
  label: "My team",
  teamName: null,
  expiresAt: null,
  isExpired: false,
};

const CURRENT: MarsSession = {
  session_id: "sess_current",
  name: "fix-login",
  status: "SESSION_STATUS_READY",
  config_id: "cfg_1",
};

const SIBLING: MarsSession = {
  session_id: "sess_sibling",
  name: "add-billing",
  status: "SESSION_STATUS_PAUSED",
  config_id: "cfg_1",
};

const BACKEND: Backend = {
  id: "mars-sess_current",
  name: "Web agent · fix-login",
  kind: "local",
  host: "http://127.0.0.1:51000",
  apiKey: "",
  marsSessionId: CURRENT.session_id,
  marsConfigId: "cfg_1",
};

function fakeBridge() {
  return {
    getAuthState: vi.fn(async () => ({
      connections: [CONNECTION],
      active: CONNECTION,
      isPersistent: true,
      canUseOAuth: false,
    })),
    listAgentConfigs: vi.fn(async () => ({
      configs: [{ id: "cfg_1", name: "Web agent", agent: "openhands" }],
      nextPageToken: null,
    })),
    listSessions: vi.fn(async () => ({ sessions: [], nextPageToken: null })),
    listConfigSessions: vi.fn(async () => ({
      sessions: [CURRENT, SIBLING],
      nextPageToken: null,
    })),
    createSession: vi.fn(async () => ({
      session_id: "sess_new",
      name: "fresh",
      status: "SESSION_STATUS_READY",
      config_id: "cfg_1",
    })),
    openTunnel: vi.fn(async ({ sessionId }: { sessionId: string }) => ({
      sessionId,
      status: "connected" as const,
      remotePort: 8000,
      localPort: 51001,
      error: undefined,
    })),
    closeTunnel: vi.fn(async () => {}),
    getTunnel: vi.fn(async () => undefined),
  } as unknown as MarsBridge;
}

function renderPanel() {
  const navigate = vi.fn();
  renderWithProviders(
    <ActiveBackendProvider>
      <MarsSessionPanel backend={BACKEND} />
    </ActiveBackendProvider>,
    { navigation: { navigate } },
  );
  return { navigate };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
});

afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
  delete window.marsBridge;
});

describe("MarsSessionPanel", () => {
  it("lists the agent's sessions and opens another one with a click", async () => {
    const bridge = fakeBridge();
    window.marsBridge = bridge;
    const { navigate } = renderPanel();

    const current = await screen.findByTestId("mars-session-sess_current");
    expect(
      within(current).getByTestId("mars-session-open-sess_current"),
    ).toHaveAttribute("aria-current", "true");
    await userEvent.click(
      await screen.findByTestId("mars-session-open-sess_sibling"),
    );

    await waitFor(() => expect(navigate).toHaveBeenCalled());
    expect(bridge.openTunnel).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId: "sess_sibling" }),
    );
  });

  it("launches a new session for the current agent and connects to it", async () => {
    const bridge = fakeBridge();
    window.marsBridge = bridge;
    const { navigate } = renderPanel();

    await userEvent.click(await screen.findByTestId("mars-session-panel-new"));

    await waitFor(() => expect(navigate).toHaveBeenCalled());
    expect(bridge.createSession).toHaveBeenCalledWith(
      "cfg_1",
      expect.any(String),
    );
    expect(bridge.openTunnel).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId: "sess_new" }),
    );
  });
});
