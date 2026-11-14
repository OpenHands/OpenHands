import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  useHomeDirectory,
  useSearchSubdirs,
} from "#/hooks/query/use-search-subdirs";

// The active backend is swapped per-test so a connection edit can be modelled
// the way the context does it: same backend id, bumped connectionRevision.
const backendMock = vi.hoisted(() => ({
  current: {
    backend: {
      id: "local-1",
      kind: "local" as "local" | "cloud",
      connectionRevision: 0,
    },
    orgId: null as string | null,
  },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => backendMock.current,
}));

const { mockGetHome, mockSearchSubdirectories } = vi.hoisted(() => ({
  mockGetHome: vi.fn(),
  mockSearchSubdirectories: vi.fn(),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: () => ({}),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  FileClient: vi.fn(function FileClientMock() {
    return {
      getHome: mockGetHome,
      searchSubdirectories: mockSearchSubdirectories,
    };
  }),
}));

// One client per test so the cache is visible across rerenders, which is the
// whole point: the second mount must not be served the first connection's data.
function makeWrapper(client: QueryClient) {
  return function wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

const newClient = () =>
  new QueryClient({ defaultOptions: { queries: { retry: false } } });

beforeEach(() => {
  vi.clearAllMocks();
  backendMock.current = {
    backend: { id: "local-1", kind: "local", connectionRevision: 0 },
    orgId: null,
  };
});

describe("useHomeDirectory", () => {
  it("refetches HOME after a connection edit that keeps the backend id", async () => {
    const client = newClient();
    mockGetHome
      .mockResolvedValueOnce({ home: "/home/first-server" })
      .mockResolvedValueOnce({ home: "/home/second-server" });

    const { result, rerender } = renderHook(() => useHomeDirectory(), {
      wrapper: makeWrapper(client),
    });

    await waitFor(() =>
      expect(result.current.data?.home).toBe("/home/first-server"),
    );

    // Editing host/API key bumps the revision but keeps the backend id.
    backendMock.current = {
      backend: { id: "local-1", kind: "local", connectionRevision: 1 },
      orgId: null,
    };
    rerender();

    await waitFor(() =>
      expect(result.current.data?.home).toBe("/home/second-server"),
    );
    expect(mockGetHome).toHaveBeenCalledTimes(2);
  });

  it("keeps the cached HOME when only the display name changes", async () => {
    const client = newClient();
    mockGetHome.mockResolvedValue({ home: "/home/server" });

    const { result, rerender } = renderHook(() => useHomeDirectory(), {
      wrapper: makeWrapper(client),
    });

    await waitFor(() => expect(result.current.data?.home).toBe("/home/server"));

    // A rename touches neither host nor API key, so the revision is unchanged.
    backendMock.current = {
      backend: { id: "local-1", kind: "local", connectionRevision: 0 },
      orgId: null,
    };
    rerender();

    await waitFor(() => expect(result.current.data?.home).toBe("/home/server"));
    expect(mockGetHome).toHaveBeenCalledTimes(1);
  });

  it("still refetches when the backend id changes", async () => {
    const client = newClient();
    mockGetHome
      .mockResolvedValueOnce({ home: "/home/a" })
      .mockResolvedValueOnce({ home: "/home/b" });

    const { result, rerender } = renderHook(() => useHomeDirectory(), {
      wrapper: makeWrapper(client),
    });

    await waitFor(() => expect(result.current.data?.home).toBe("/home/a"));

    backendMock.current = {
      backend: { id: "local-2", kind: "local", connectionRevision: 0 },
      orgId: null,
    };
    rerender();

    await waitFor(() => expect(result.current.data?.home).toBe("/home/b"));
  });
});

describe("useSearchSubdirs", () => {
  it("refetches the listing after a connection edit that keeps the backend id", async () => {
    const client = newClient();
    mockSearchSubdirectories
      .mockResolvedValueOnce({
        items: [{ name: "first-home-marker", path: "/home/first-home-marker" }],
        next_page_id: null,
      })
      .mockResolvedValueOnce({
        items: [
          { name: "second-home-marker", path: "/home/second-home-marker" },
        ],
        next_page_id: null,
      });

    const { result, rerender } = renderHook(
      () => useSearchSubdirs("/home"),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() =>
      expect(result.current.data?.items[0].name).toBe("first-home-marker"),
    );

    backendMock.current = {
      backend: { id: "local-1", kind: "local", connectionRevision: 1 },
      orgId: null,
    };
    rerender();

    await waitFor(() =>
      expect(result.current.data?.items[0].name).toBe("second-home-marker"),
    );
    expect(mockSearchSubdirectories).toHaveBeenCalledTimes(2);
  });

  it("does not fetch without a path", () => {
    const client = newClient();
    renderHook(() => useSearchSubdirs(null), { wrapper: makeWrapper(client) });

    expect(mockSearchSubdirectories).not.toHaveBeenCalled();
  });
});
