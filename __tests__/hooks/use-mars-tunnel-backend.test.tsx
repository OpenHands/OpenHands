import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("#/services/telemetry", () => ({
  setTelemetryCloudContext: vi.fn(),
  setTelemetryIdentity: vi.fn(),
}));

import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { __resetHealthStoreForTests } from "#/api/backend-registry/health-store";
import {
  ActiveBackendProvider,
  useActiveBackendContext,
} from "#/contexts/active-backend-context";
import { useMarsTunnelBackend } from "#/hooks/use-mars-tunnel-backend";
import type { MarsTunnelStatus } from "#/api/mars/mars-tunnel-backend";

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

function fakeMarsBridge() {
  const tunnels = new Map<string, MarsTunnelStatus>();
  return {
    openTunnel: vi.fn(async (params: { sessionId: string; remotePort: number }) => {
      const status: MarsTunnelStatus = {
        sessionId: params.sessionId,
        status: "connected",
        remotePort: params.remotePort,
        localPort: 51000 + tunnels.size,
        error: undefined,
      };
      tunnels.set(params.sessionId, status);
      return status;
    }),
    closeTunnel: vi.fn(async (sessionId: string) => {
      tunnels.delete(sessionId);
    }),
    getTunnel: vi.fn(async (sessionId: string) => tunnels.get(sessionId)),
  };
}

beforeEach(() => {
  window.localStorage.clear();
  vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
  vi.stubEnv("VITE_SESSION_API_KEY", "session-key");
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
  window.marsBridge = fakeMarsBridge();
});

afterEach(() => {
  window.localStorage.clear();
  vi.unstubAllEnvs();
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
  delete window.marsBridge;
});

describe("useMarsTunnelBackend", () => {
  it("attach() opens the tunnel and registers the result as a local Backend", async () => {
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });

    let backend: { id: string } | undefined;
    await act(async () => {
      backend = await result.current.tunnel.attach({
        sessionId: "sess_abc",
        remotePort: 8000,
        accessToken: "token",
        name: "My MARS session",
      });
    });

    expect(window.marsBridge!.openTunnel).toHaveBeenCalledWith({
      sessionId: "sess_abc",
      remotePort: 8000,
      accessToken: "token",
    });
    expect(
      result.current.active.backends.find((b) => b.id === backend!.id),
    ).toMatchObject({
      name: "My MARS session",
      host: "http://127.0.0.1:51000",
      kind: "local",
      authMode: "api-key",
    });
  });

  it("attach() auto-switches the active backend to the newly attached session", async () => {
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });

    await act(async () => {
      await result.current.tunnel.attach({
        sessionId: "sess_abc",
        remotePort: 8000,
        accessToken: "token",
        name: "My MARS session",
      });
    });

    expect(result.current.active.active.backend.name).toBe("My MARS session");
  });

  it("attach() throws and registers nothing when the tunnel fails to open", async () => {
    window.marsBridge!.openTunnel = vi.fn(async () => ({
      sessionId: "sess_bad",
      status: "error" as const,
      remotePort: 8000,
      localPort: undefined,
      error: "server rejected tunnel (403 Forbidden): invalid token",
    }));
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });
    const backendCountBefore = result.current.active.backends.length;

    await expect(
      act(() =>
        result.current.tunnel.attach({
          sessionId: "sess_bad",
          remotePort: 8000,
          accessToken: "wrong",
          name: "Bad session",
        }),
      ),
    ).rejects.toThrow(/invalid token/);

    expect(result.current.active.backends).toHaveLength(backendCountBefore);
  });

  it("detach() closes the tunnel and removes the Backend", async () => {
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });

    let backend: { id: string } | undefined;
    await act(async () => {
      backend = await result.current.tunnel.attach({
        sessionId: "sess_abc",
        remotePort: 8000,
        accessToken: "token",
        name: "My MARS session",
      });
    });

    await act(async () => {
      await result.current.tunnel.detach(backend!.id, "sess_abc");
    });

    expect(window.marsBridge!.closeTunnel).toHaveBeenCalledWith("sess_abc");
    expect(
      result.current.active.backends.find((b) => b.id === backend!.id),
    ).toBeUndefined();
  });

  it("detach() does not remove the Backend if closing the tunnel fails", async () => {
    const { result } = renderHook(useCombined, { wrapper: makeWrapper() });

    let backend: { id: string } | undefined;
    await act(async () => {
      backend = await result.current.tunnel.attach({
        sessionId: "sess_abc",
        remotePort: 8000,
        accessToken: "token",
        name: "My MARS session",
      });
    });

    window.marsBridge!.closeTunnel = vi.fn(async () => {
      throw new Error("IPC failure");
    });

    await expect(
      act(() => result.current.tunnel.detach(backend!.id, "sess_abc")),
    ).rejects.toThrow(/IPC failure/);

    expect(
      result.current.active.backends.find((b) => b.id === backend!.id),
    ).toBeDefined();
  });
});
