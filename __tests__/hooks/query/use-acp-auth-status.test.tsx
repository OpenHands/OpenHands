import React from "react";
import {
  QueryClient,
  QueryClientProvider,
  focusManager,
} from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useAcpAuthStatus } from "#/hooks/query/use-acp-auth-status";

// Active backend is swapped per-test (local vs cloud) via this mutable holder.
const backendMock = vi.hoisted(() => ({
  current: {
    backend: { id: "local-1", kind: "local" as "local" | "cloud" },
    orgId: null as string | null,
  },
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => backendMock.current,
}));

// AcpService.getAuthStatus resolves an AcpAuthStatus string (it runs the
// detection command + classifies internally; that logic is unit-tested in
// acp-service.api.test.ts).
const getAuthStatus = vi.hoisted(() => vi.fn());
vi.mock("#/api/acp-service/acp-service.api", () => ({
  default: {
    getAuthStatus: (...args: unknown[]) => getAuthStatus(...args),
  },
}));

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  vi.clearAllMocks();
  backendMock.current = {
    backend: { id: "local-1", kind: "local" },
    orgId: null,
  };
});

describe("useAcpAuthStatus", () => {
  it("reports authenticated when the probe says so", async () => {
    getAuthStatus.mockResolvedValue("authenticated");

    const { result } = renderHook(() => useAcpAuthStatus("claude-code"), {
      wrapper,
    });

    await waitFor(() => expect(result.current.status).toBe("authenticated"));
    // The provider key parameterizes the probe.
    expect(getAuthStatus).toHaveBeenCalledTimes(1);
    expect(getAuthStatus).toHaveBeenCalledWith("claude-code");
  });

  it("reports unauthenticated when the probe says so", async () => {
    getAuthStatus.mockResolvedValue("unauthenticated");

    const { result } = renderHook(() => useAcpAuthStatus("codex"), { wrapper });

    await waitFor(() => expect(result.current.status).toBe("unauthenticated"));
    expect(getAuthStatus).toHaveBeenCalledWith("codex");
  });

  it("falls back to unknown when the probe call rejects", async () => {
    getAuthStatus.mockRejectedValue(new Error("network down"));

    const { result } = renderHook(() => useAcpAuthStatus("claude-code"), {
      wrapper,
    });

    // A rejected probe must not be read as "not logged in" — it stays unknown
    // so the caller keeps showing the API-key fields.
    await waitFor(() => expect(result.current.isChecking).toBe(false));
    expect(result.current.status).toBe("unknown");
  });

  it("surfaces an unknown status from the probe verbatim", async () => {
    getAuthStatus.mockResolvedValue("unknown");

    const { result } = renderHook(() => useAcpAuthStatus("claude-code"), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isChecking).toBe(false));
    expect(result.current.status).toBe("unknown");
  });

  it("does not probe on a cloud backend", async () => {
    backendMock.current = {
      backend: { id: "cloud-1", kind: "cloud" },
      orgId: null,
    };

    const { result } = renderHook(() => useAcpAuthStatus("claude-code"), {
      wrapper,
    });

    await Promise.resolve();
    expect(result.current.status).toBe("unknown");
    expect(result.current.isSupported).toBe(false);
    // The probe is gated off, so it never enters a "checking" state — the
    // banner consumer (resolveAcpAuthDisplay) can fall straight through to the
    // credentials-configured signal instead of spinning on a probe that will
    // never run. This is the behaviour the Docker/cloud banner state relies on.
    expect(result.current.isChecking).toBe(false);
    expect(getAuthStatus).not.toHaveBeenCalled();
  });

  it("probes credential-less providers too (e.g. gemini-cli, OAuth login)", async () => {
    // Eligibility is not tied to having API-key fields — the server can detect
    // subscription/OAuth providers like Gemini, so the hook must still probe.
    getAuthStatus.mockResolvedValue("authenticated");

    const { result } = renderHook(() => useAcpAuthStatus("gemini-cli"), {
      wrapper,
    });

    await waitFor(() => expect(result.current.status).toBe("authenticated"));
    expect(result.current.isSupported).toBe(true);
    expect(getAuthStatus).toHaveBeenCalledWith("gemini-cli");
  });

  describe("when the login changes after the first probe", () => {
    beforeEach(() => {
      // Fake only Date so react-query's staleness clock can be advanced while
      // promises and waitFor keep running on real timers.
      vi.useFakeTimers({ toFake: ["Date"] });
      // Queued one-shot probe results must not leak between cases.
      getAuthStatus.mockReset();
    });
    afterEach(() => {
      focusManager.setFocused(undefined);
      vi.useRealTimers();
    });

    it("re-probes when the window regains focus, keeping the last result visible meanwhile", async () => {
      getAuthStatus.mockResolvedValueOnce("unauthenticated");
      const { result } = renderHook(() => useAcpAuthStatus("claude-code"), {
        wrapper,
      });
      await waitFor(() =>
        expect(result.current.status).toBe("unauthenticated"),
      );

      // The user signs in to the CLI in a terminal, then comes back.
      let resolveReprobe!: (status: string) => void;
      getAuthStatus.mockReturnValueOnce(
        new Promise((resolve) => {
          resolveReprobe = resolve;
        }),
      );
      vi.advanceTimersByTime(60_000);
      act(() => {
        focusManager.setFocused(false);
        focusManager.setFocused(true);
      });

      await waitFor(() => expect(getAuthStatus).toHaveBeenCalledTimes(2));
      // A background re-check must not flash the "checking" state or drop
      // the previous answer.
      expect(result.current.isChecking).toBe(false);
      expect(result.current.status).toBe("unauthenticated");

      act(() => resolveReprobe("authenticated"));
      await waitFor(() => expect(result.current.status).toBe("authenticated"));
    });

    it("re-probes when the consuming surface mounts again", async () => {
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      });
      const sharedWrapper = ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      );
      getAuthStatus.mockResolvedValueOnce("unauthenticated");
      const first = renderHook(() => useAcpAuthStatus("claude-code"), {
        wrapper: sharedWrapper,
      });
      await waitFor(() =>
        expect(first.result.current.status).toBe("unauthenticated"),
      );
      first.unmount();

      // The user leaves Settings, signs in, and comes back to the page.
      getAuthStatus.mockResolvedValueOnce("authenticated");
      vi.advanceTimersByTime(60_000);
      const second = renderHook(() => useAcpAuthStatus("claude-code"), {
        wrapper: sharedWrapper,
      });

      await waitFor(() =>
        expect(second.result.current.status).toBe("authenticated"),
      );
      expect(getAuthStatus).toHaveBeenCalledTimes(2);
    });
  });

  it("does not probe when disabled (e.g. the step is not the active slide)", async () => {
    const { result } = renderHook(
      () => useAcpAuthStatus("claude-code", { enabled: false }),
      { wrapper },
    );

    await Promise.resolve();
    expect(result.current.status).toBe("unknown");
    expect(getAuthStatus).not.toHaveBeenCalled();
  });
});
