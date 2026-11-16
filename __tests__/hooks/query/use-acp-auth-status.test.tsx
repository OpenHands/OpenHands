import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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

  it("does not probe when disabled (e.g. the step is not the active slide)", async () => {
    const { result } = renderHook(
      () => useAcpAuthStatus("claude-code", { enabled: false }),
      { wrapper },
    );

    await Promise.resolve();
    expect(result.current.status).toBe("unknown");
    expect(getAuthStatus).not.toHaveBeenCalled();
  });
  it("re-probes when the surface remounts, so CLI sign-in shows up", async () => {
    // #19730: the status used to be cached for the whole session; a user who
    // signs in via the CLI and re-opens Settings must see the fresh state.
    getAuthStatus.mockResolvedValue("unauthenticated");
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const first = renderHook(() => useAcpAuthStatus("claude-code"), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    });
    await waitFor(() => expect(first.result.current.status).toBe("unauthenticated"));
    first.unmount();

    // The user signs in via the CLI outside Canvas; the probe now says so.
    getAuthStatus.mockResolvedValue("authenticated");
    const second = renderHook(() => useAcpAuthStatus("claude-code"), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    });
    await waitFor(() => expect(second.result.current.status).toBe("authenticated"));
    expect(getAuthStatus).toHaveBeenCalledTimes(2);
  });

  it("re-probes on window focus and keeps the previous status visible while refetching", async () => {
    getAuthStatus.mockResolvedValue("unauthenticated");
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const { result } = renderHook(() => useAcpAuthStatus("claude-code"), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    });
    await waitFor(() => expect(result.current.status).toBe("unauthenticated"));

    // Sign in via the CLI, then return focus to the Canvas tab: the probe
    // re-runs, and the stale "unauthenticated" stays on screen (no flash of
    // the initial checking state) until the fresh status lands.
    getAuthStatus.mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(() => resolve("authenticated"), 50),
        ),
    );
    await act(async () => {
      // React Query's focus manager listens for focus/visibility changes.
      Object.defineProperty(document, "visibilityState", {
        configurable: true,
        value: "visible",
      });
      // React Query v5's focus manager listens for `visibilitychange` on
      // window (jsdom never fires it for real focus changes).
      window.dispatchEvent(new Event("visibilitychange"));
    });
    expect(result.current.status).toBe("unauthenticated");
    expect(result.current.isChecking).toBe(false);
    await waitFor(() => expect(result.current.status).toBe("authenticated"));
    expect(getAuthStatus).toHaveBeenCalledTimes(2);
  });
});
