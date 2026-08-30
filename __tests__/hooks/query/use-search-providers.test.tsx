import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { callCloudProxy } from "#/api/cloud/proxy";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import { server } from "#/mocks/node";

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
};

const cloudBackend: Backend = {
  id: "cloud-ohe",
  name: "OpenHands Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-key",
  kind: "cloud",
};

describe("useSearchProviders — local backend", () => {
  it("returns providers that sort past the first 100 entries", async () => {
    // Arrange: mirror the real local agent-server, whose litellm-derived
    // provider list is ~150 entries long and sorted alphabetically, putting
    // "openrouter" at index 101, past any 100-item cap.
    const providers = [
      ...Array.from(
        { length: 101 },
        (_, i) => `provider_${String(i).padStart(3, "0")}`,
      ),
      "openrouter",
      ...Array.from({ length: 47 }, (_, i) => `zprovider_${i}`),
    ];
    server.use(
      http.get("/api/llm/providers", () => HttpResponse.json({ providers })),
      http.get("/api/llm/models/verified", () =>
        HttpResponse.json({ models: { openhands: ["claude-opus-4-7"] } }),
      ),
    );

    // Act
    const { result } = renderHook(() => useSearchProviders(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // Assert: the picker must surface every provider the backend reports.
    const names = result.current.data?.map((provider) => provider.name) ?? [];
    expect(names).toContain("openrouter");
    expect(names).toHaveLength(providers.length + 1); // + the verified "openhands"
  });
});

describe("useSearchProviders — cloud backend pagination", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: null });
    vi.mocked(callCloudProxy).mockReset();
  });

  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    vi.mocked(callCloudProxy).mockReset();
  });

  it("follows next_page_id until exhaustion on the cloud backend so providers past the default page size still appear", async () => {
    // Arrange: cloud service paginates by default — page 1 holds 100 entries
    // (the live app.all-hands.dev screenshot showed only the fuzzy match for
    // "xai" because the real xAI entry sorted past the cut). Page 2 carries
    // the rest, including "xai" and "openrouter".
    const allProviders = [
      ...Array.from(
        { length: 100 },
        (_, i) => `provider_${String(i).padStart(3, "0")}`,
      ),
      "openrouter",
      "xai",
      ...Array.from({ length: 47 }, (_, i) => `zprovider_${i}`),
    ];
    const page1Items = allProviders.slice(0, 100).map((name) => ({
      name,
      verified: false,
    }));
    const page2Items = allProviders.slice(100).map((name) => ({
      name,
      verified: false,
    }));

    vi.mocked(callCloudProxy).mockImplementation((async (req: {
      path: string;
      method: string;
    }) => {
      expect(req.method).toBe("GET");
      expect(req.path).toMatch(/^\/api\/v1\/config\/providers\/search/);
      const url = new URL(`http://x.example.com${req.path}`);
      const pageId = url.searchParams.get("page_id");
      if (!pageId) {
        return { items: page1Items, next_page_id: "page-2" };
      }
      if (pageId === "page-2") {
        return { items: page2Items, next_page_id: null };
      }
      throw new Error(`Unexpected page_id ${pageId}`);
    }) as never);

    // Act
    const { result } = renderHook(() => useSearchProviders(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // Assert: both "openrouter" (sorts past index 100) and "xai" (the live
    // bug report) must surface, and the requested page_id must be plumbed
    // through to the cloud proxy on the second call.
    const names = result.current.data?.map((provider) => provider.name) ?? [];
    expect(names).toContain("openrouter");
    expect(names).toContain("xai");
    expect(names).toHaveLength(allProviders.length);

    const calls = vi.mocked(callCloudProxy).mock.calls;
    expect(calls.length).toBeGreaterThanOrEqual(2);
    const paths = calls.map((c) => (c[0] as { path: string }).path);
    expect(paths[0]).not.toMatch(/[?&]page_id=/);
    expect(paths[1]).toMatch(/[?&]page_id=page-2/);
  });

  it("throws instead of looping forever when the cloud backend returns a repeated next_page_id", async () => {
    // Arrange: the cloud service is out of our control — a buggy cursor
    // must not hang the settings page. Force a 2-cycle so the cycle guard
    // has to fire on the third request.
    vi.mocked(callCloudProxy).mockImplementation((async () => ({
      items: [{ name: "stuck-provider", verified: false }],
      next_page_id: "page-loop",
    })) as never);

    // Act
    const { result } = renderHook(() => useSearchProviders(), { wrapper });

    // Assert: the hook surfaces the error instead of hanging. We bound the
    // wait so a regression that drops the guard would fail the test rather
    // than stall the suite.
    await waitFor(() => expect(result.current.isError).toBe(true), {
      timeout: 2_000,
    });
    expect(result.current.error).toBeInstanceOf(Error);
    expect(String(result.current.error?.message)).toMatch(
      /Repeated page id|Too many pagination/,
    );
  });
});
