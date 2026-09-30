import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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
import {
  ACTIVE_BACKEND_STORAGE_KEY,
  BACKENDS_STORAGE_KEY,
} from "#/api/backend-registry/storage";
import {
  ActiveBackendProvider,
  useActiveBackendContext,
} from "#/contexts/active-backend-context";
import {
  useMarsTunnelBackend,
  useRestoreMarsTunnels,
} from "#/hooks/use-mars-tunnel-backend";
import type {
  MarsBridge,
  MarsTunnelStatus,
  OpenMarsTunnelParams,
} from "#/api/mars/mars-tunnel-backend";
import type { Backend } from "#/api/backend-registry/types";

function makeWrapper() {
  const queryClient = new QueryClient();
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  }
  return Wrapper;
}

function useCombined() {
  return {
    tunnel: useMarsTunnelBackend(),
    active: useActiveBackendContext(),
  };
}

function tunnelStatus(
  sessionId: string,
  localPort: number | undefined,
  error?: string,
): MarsTunnelStatus {
  return {
    sessionId,
    status: error ? "error" : "connected",
    remotePort: 8000,
    localPort,
    error,
  };
}

function fakeMarsBridge(localPort = 51000) {
  return {
    openTunnel: vi.fn(async ({ sessionId }: OpenMarsTunnelParams) =>
      tunnelStatus(sessionId, localPort),
    ),
    closeTunnel: vi.fn(async () => {}),
    getTunnel: vi.fn(async () => undefined),
  } as unknown as MarsBridge & {
    openTunnel: ReturnType<typeof vi.fn>;
    closeTunnel: ReturnType<typeof vi.fn>;
  };
}

const SESSION = { sessionId: "sess_abc", name: "Agent · sess-1" };

beforeEach(() => {
  window.localStorage.clear();
  vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
  vi.stubEnv("VITE_SESSION_API_KEY", "session-key");
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
  validateLocalBackend.mockResolvedValue("1.44.1");
  searchConversations.mockResolvedValue({ items: [] });
  window.marsBridge = fakeMarsBridge();
});

afterEach(() => {
  window.localStorage.clear();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
  delete window.marsBridge;
});

describe("useMarsTunnelBackend", () => {
  it("attach() registers the healthy tunnel as the active local backend for that session", async () => {
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });

    await act(async () => {
      await result.current.tunnel.attach({ ...SESSION, configId: "cfg_1" });
    });

    expect(window.marsBridge!.openTunnel).toHaveBeenCalledWith({
      sessionId: "sess_abc",
      localPort: undefined,
    });
    expect(result.current.active.active.backend).toMatchObject({
      name: "Agent · sess-1",
      host: "http://127.0.0.1:51000",
      kind: "local",
      marsSessionId: "sess_abc",
      marsConfigId: "cfg_1",
    });
  });

  it("attach() re-points the session's existing backend instead of adding a duplicate", async () => {
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });
    await act(async () => {
      await result.current.tunnel.attach(SESSION);
    });
    const countAfterFirst = result.current.active.backends.length;
    window.marsBridge = fakeMarsBridge(52000);

    await act(async () => {
      await result.current.tunnel.attach(SESSION);
    });

    expect(result.current.active.backends).toHaveLength(countAfterFirst);
    expect(
      result.current.active.backends.find((b) => b.marsSessionId === "sess_abc")
        ?.host,
    ).toBe("http://127.0.0.1:52000");
  });

  it("attach() throws and registers nothing when the tunnel fails to open", async () => {
    window.marsBridge!.openTunnel = vi.fn(async () =>
      tunnelStatus("sess_abc", undefined, "server rejected tunnel (403)"),
    );
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });
    const countBefore = result.current.active.backends.length;

    await expect(
      act(() => result.current.tunnel.attach(SESSION)),
    ).rejects.toThrow(/403/);

    expect(result.current.active.backends).toHaveLength(countBefore);
  });

  it("attach() closes the tunnel when the agent-server behind it never answers", async () => {
    validateLocalBackend.mockRejectedValue(new Error("connection refused"));
    // Expire the 60s probe budget after the first failed attempt.
    vi.spyOn(Date, "now")
      .mockReturnValueOnce(0)
      .mockReturnValue(Number.MAX_SAFE_INTEGER);
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });
    const countBefore = result.current.active.backends.length;

    await expect(
      act(() => result.current.tunnel.attach(SESSION)),
    ).rejects.toThrow(/connection refused/);

    expect(window.marsBridge!.closeTunnel).toHaveBeenCalledWith("sess_abc");
    expect(result.current.active.backends).toHaveLength(countBefore);
  });

  it("attach() reports the session's most recent conversation so the caller can land in it", async () => {
    searchConversations.mockResolvedValueOnce({
      items: [{ id: "conv_latest" }],
    });
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });

    let attached: Awaited<ReturnType<typeof result.current.tunnel.attach>>;
    await act(async () => {
      attached = await result.current.tunnel.attach(SESSION);
    });

    expect(attached!.latestConversationId).toBe("conv_latest");
  });

  it("attach() fails fast with a not-openhands error when the guest port stays closed", async () => {
    validateLocalBackend.mockRejectedValue(new Error("socket hang up"));
    window.marsBridge!.getTunnel = vi.fn(async () => ({
      ...tunnelStatus("sess_abc", 51000),
      upstreamFailure: { closeCode: 4002, httpStatus: null, message: "" },
    }));
    // First failure starts the grace window; the next is past it but still
    // well inside the 60s probe budget.
    vi.spyOn(Date, "now")
      .mockReturnValueOnce(0)
      .mockReturnValueOnce(0)
      .mockReturnValue(25_000);
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });

    await expect(
      act(() => result.current.tunnel.attach(SESSION)),
    ).rejects.toMatchObject({
      name: "MarsAttachError",
      reason: "not-openhands",
    });
    expect(window.marsBridge!.closeTunnel).toHaveBeenCalledWith("sess_abc");
  });

  it("detach() closes the tunnel and removes the Backend", async () => {
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });
    let backend: Backend;
    await act(async () => {
      ({ backend } = await result.current.tunnel.attach(SESSION));
    });

    await act(async () => {
      await result.current.tunnel.detach(backend!);
    });

    expect(window.marsBridge!.closeTunnel).toHaveBeenCalledWith("sess_abc");
    expect(
      result.current.active.backends.find((b) => b.id === backend!.id),
    ).toBeUndefined();
  });

  it("detach() keeps the Backend if closing the tunnel fails", async () => {
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });
    let backend: Backend;
    await act(async () => {
      ({ backend } = await result.current.tunnel.attach(SESSION));
    });
    window.marsBridge!.closeTunnel = vi.fn(async () => {
      throw new Error("IPC failure");
    });

    await expect(
      act(() => result.current.tunnel.detach(backend!)),
    ).rejects.toThrow(/IPC failure/);

    expect(
      result.current.active.backends.find((b) => b.id === backend!.id),
    ).toBeDefined();
  });
});

describe("useRestoreMarsTunnels", () => {
  it("re-opens only the active session's tunnel on its previous port and follows a port change", async () => {
    // Dialing a session counts as activity, so restoring every persisted
    // session would keep idle sandboxes awake and billing.
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([
        {
          id: "mars-idle",
          name: "Agent · sess-0",
          host: "http://127.0.0.1:50000",
          apiKey: "",
          kind: "local",
          marsSessionId: "sess_idle",
        },
        {
          id: "mars-1",
          name: "Agent · sess-1",
          host: "http://127.0.0.1:51000",
          apiKey: "",
          kind: "local",
          marsSessionId: "sess_abc",
        },
      ]),
    );
    window.localStorage.setItem(
      ACTIVE_BACKEND_STORAGE_KEY,
      JSON.stringify({ backendId: "mars-1", orgId: null }),
    );
    __resetActiveStoreForTests();
    window.marsBridge = fakeMarsBridge(52000);

    const { result } = renderHook(
      () => {
        useRestoreMarsTunnels();
        return useActiveBackendContext();
      },
      { wrapper: makeWrapper() },
    );

    await waitFor(() =>
      expect(result.current.backends.find((b) => b.id === "mars-1")?.host).toBe(
        "http://127.0.0.1:52000",
      ),
    );
    expect(window.marsBridge!.openTunnel).toHaveBeenCalledTimes(1);
    expect(window.marsBridge!.openTunnel).toHaveBeenCalledWith({
      sessionId: "sess_abc",
      localPort: 51000,
    });
  });

  it("reports the launch restore as pending until the session's agent-server answers", async () => {
    // Arrange — the active session's sandbox is still booting after launch.
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([
        {
          id: "mars-1",
          name: "Agent · sess-1",
          host: "http://127.0.0.1:51000",
          apiKey: "",
          kind: "local",
          marsSessionId: "sess_abc",
        },
      ]),
    );
    window.localStorage.setItem(
      ACTIVE_BACKEND_STORAGE_KEY,
      JSON.stringify({ backendId: "mars-1", orgId: null }),
    );
    __resetActiveStoreForTests();
    let bootAgentServer: (version: string) => void = () => {};
    validateLocalBackend.mockReturnValueOnce(
      new Promise((resolve) => {
        bootAgentServer = resolve;
      }),
    );

    // Act
    const { result } = renderHook(() => useRestoreMarsTunnels(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(validateLocalBackend).toHaveBeenCalled());
    const pendingWhileBooting = result.current;
    act(() => bootAgentServer("1.44.1"));

    // Assert
    expect(pendingWhileBooting).toBe(true);
    await waitFor(() => expect(result.current).toBe(false));
  });
});
