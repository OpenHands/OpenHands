import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Backend } from "#/api/backend-registry/types";
import {
  searchConversationContent,
  isContentSearchEnabled,
} from "#/api/conversation-service/content-search.api";
import { useConversationContentSearch } from "#/hooks/use-conversation-content-search";

const activeBackend = vi.hoisted(() => ({ current: null as Backend | null }));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: activeBackend.current, orgId: null }),
}));

const local: Backend = {
  id: "local",
  name: "local",
  host: "http://a.test",
  apiKey: "key",
  kind: "local",
};

const cloud: Backend = {
  id: "cloud",
  name: "cloud",
  host: "https://app.all-hands.dev",
  apiKey: "",
  kind: "cloud",
};

vi.mock("#/api/conversation-service/content-search.api", () => ({
  searchConversationContent: vi.fn(),
  isContentSearchEnabled: vi.fn(),
}));

function wrapper({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider
      client={
        new QueryClient({
          defaultOptions: { queries: { retry: false, gcTime: 0 } },
        })
      }
    >
      {children}
    </QueryClientProvider>
  );
}

const searchMock = vi.mocked(searchConversationContent);
const probeMock = vi.mocked(isContentSearchEnabled);

beforeEach(() => {
  vi.restoreAllMocks();
  // The probe gates searching; default it to "on" so the search-focused
  // tests exercise the query path. Individual tests override this.
  probeMock.mockResolvedValue(true);
  vi.mocked(searchConversationContent).mockImplementation(
    vi.fn(async (query: string, _limit?: number) => ({
      results: [
        {
          conversation_id: `conv-for-${query}`,
          title: "A conversation",
          event_id: "event-1",
          source: "user",
          timestamp: "2026-11-01T00:00:00Z",
          snippet: "the [wombat] climbed",
        },
      ],
      total: 1,
      capped: false,
    })),
  );
  activeBackend.current = local;
});

afterEach(() => {
  vi.useRealTimers();
});

it("does not search until the user types", () => {
  const { result } = renderHook(() => useConversationContentSearch(), {
    wrapper,
  });
  expect(result.current.hits).toEqual([]);
  expect(result.current.isSearching).toBe(false);
  expect(searchMock).not.toHaveBeenCalled();
});

it("debounces keystrokes before hitting the API", async () => {
  const { result } = renderHook(() => useConversationContentSearch(), {
    wrapper,
  });

  act(() => result.current.setQuery("w"));
  act(() => result.current.setQuery("wo"));
  act(() => result.current.setQuery("wom"));
  expect(searchMock).not.toHaveBeenCalled();

  // Real timers: the debounce itself is the timing under test.
  await waitFor(() =>
    expect(result.current.hits[0]?.conversation_id).toBe("conv-for-wom"),
  );
  // One call for the final debounced value only.
  expect(searchMock).toHaveBeenCalledTimes(1);
  expect(searchMock).toHaveBeenCalledWith("wom", 50);
});

it("never queries for cloud backends", async () => {
  activeBackend.current = cloud;
  vi.useFakeTimers();
  const { result } = renderHook(() => useConversationContentSearch(), {
    wrapper,
  });

  act(() => result.current.setQuery("wombat"));
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1000);
  });

  expect(result.current.isSearching).toBe(false);
  expect(searchMock).not.toHaveBeenCalled();
  // The probe is local-only too: a cloud backend never reaches the server.
  expect(probeMock).not.toHaveBeenCalled();
});

it("whitespace-only input does not query", async () => {
  vi.useFakeTimers();
  const { result } = renderHook(() => useConversationContentSearch(), {
    wrapper,
  });

  act(() => result.current.setQuery("   "));
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1000);
  });

  expect(result.current.isSearching).toBe(false);
  expect(searchMock).not.toHaveBeenCalled();
});

it("exposes the API error state without throwing", async () => {
  vi.mocked(searchConversationContent).mockRejectedValue(
    new Error("backend gone"),
  );
  const { result } = renderHook(() => useConversationContentSearch(), {
    wrapper,
  });

  act(() => result.current.setQuery("wombat"));
  await waitFor(() => expect(result.current.error).toBe(true));
  expect(result.current.hits).toEqual([]);
});

it("reset clears the query and drops cached results", async () => {
  const { result } = renderHook(() => useConversationContentSearch(), {
    wrapper,
  });

  act(() => result.current.setQuery("wombat"));
  await waitFor(() =>
    expect(result.current.hits[0]?.conversation_id).toBe("conv-for-wombat"),
  );

  act(() => result.current.reset());
  expect(result.current.query).toBe("");
  expect(result.current.isSearching).toBe(false);
  // React Query keeps the last successful payload on the observer after
  // removeQueries; the contract that matters is that the same term refetches
  // instead of being served from cache.
  act(() => result.current.setQuery("wombat"));
  await waitFor(() => expect(searchMock).toHaveBeenCalledTimes(2));
});

it("maps the API response into the hook state", async () => {
  vi.mocked(searchConversationContent).mockResolvedValue({
    results: [
      {
        conversation_id: "conv-1",
        title: "K8s migration",
        event_id: "event-1",
        source: "title",
        timestamp: null,
        snippet: "[K8s] migration notes",
      },
    ],
    total: 42,
    capped: true,
  });
  const { result } = renderHook(() => useConversationContentSearch(), {
    wrapper,
  });

  act(() => result.current.setQuery("k8s"));
  await waitFor(() => expect(result.current.hits).toHaveLength(1));
  expect(result.current.hits[0]?.conversation_id).toBe("conv-1");
  expect(result.current.total).toBe(42);
  expect(result.current.capped).toBe(true);
  expect(result.current.isPending).toBe(false);
  expect(result.current.error).toBe(false);
});

describe("content search availability probe", () => {
  it("stays hidden while the probe is in flight", () => {
    let resolveProbe: (v: boolean) => void = () => {};
    probeMock.mockReturnValue(
      new Promise<boolean>((r) => {
        resolveProbe = r;
      }),
    );
    const { result } = renderHook(() => useConversationContentSearch(), {
      wrapper,
    });
    expect(result.current.isAvailable).toBe(false);
    act(() => resolveProbe(true));
  });

  it("reports unavailable and never searches when the probe says off", async () => {
    vi.useFakeTimers();
    probeMock.mockResolvedValue(false);
    const { result } = renderHook(() => useConversationContentSearch(), {
      wrapper,
    });
    await act(async () => {
      await vi.runOnlyPendingTimersAsync();
    });

    expect(result.current.isAvailable).toBe(false);
    expect(probeMock).toHaveBeenCalledTimes(1);

    act(() => result.current.setQuery("garage door"));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(400);
    });
    expect(searchMock).not.toHaveBeenCalled();
    expect(result.current.isSearching).toBe(false);
  });

  it("becomes available after a successful probe", async () => {
    vi.useFakeTimers();
    probeMock.mockResolvedValue(true);
    searchMock.mockResolvedValue({
      results: [],
      total: 0,
      capped: false,
    });
    const { result } = renderHook(() => useConversationContentSearch(), {
      wrapper,
    });
    await act(async () => {
      await vi.runOnlyPendingTimersAsync();
    });
    expect(result.current.isAvailable).toBe(true);

    act(() => {
      result.current.setQuery("wombat");
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(400);
    });
    expect(searchMock).toHaveBeenCalledWith("wombat", 50);
    expect(result.current.isSearching).toBe(true);
  });
});
