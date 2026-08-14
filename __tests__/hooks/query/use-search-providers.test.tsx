import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import React from "react";
import { describe, expect, it } from "vitest";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import { server } from "#/mocks/node";

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
};

describe("useSearchProviders", () => {
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
