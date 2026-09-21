import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { server } from "#/mocks/node";
import ConfigService from "#/api/config-service/config-service.api";
import { useProviderModels } from "#/hooks/query/use-provider-models";

function createWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("useProviderModels OpenRouter discovery", () => {
  it("uses and caches the complete public catalog instead of the stale backend list", async () => {
    const ids = Array.from({ length: 125 }, (_, i) => `vendor/model-${i}`);
    ids.push("vendor/Model:free");
    let requests = 0;
    let authorization: string | null = null;
    const backendSearch = vi.spyOn(ConfigService, "searchModels");
    server.use(
      http.get("https://openrouter.ai/api/v1/models", ({ request }) => {
        requests += 1;
        authorization = request.headers.get("authorization");
        return HttpResponse.json({ data: ids.map((id) => ({ id })) });
      }),
      http.get("/api/llm/models/verified", () =>
        HttpResponse.json({ models: { openrouter: ["vendor/model-1"] } }),
      ),
    );
    const wrapper = createWrapper();
    const first = renderHook(() => useProviderModels("openrouter"), {
      wrapper,
    });
    await waitFor(() =>
      expect(first.result.current.data?.map((model) => model.name)).toEqual(
        ids,
      ),
    );
    expect(
      first.result.current.data?.find(
        (model) => model.name === "vendor/model-1",
      )?.verified,
    ).toBe(true);
    first.unmount();
    const second = renderHook(() => useProviderModels("openrouter"), {
      wrapper,
    });
    await waitFor(() => expect(second.result.current.isSuccess).toBe(true));
    expect(requests).toBe(1);
    expect(authorization).toBeNull();
    expect(backendSearch).not.toHaveBeenCalled();
  });

  it.each(["unavailable", "malformed", "empty"])(
    "falls back to backend discovery when the live catalog is %s",
    async (failure) => {
      server.use(
        http.get("https://openrouter.ai/api/v1/models", () => {
          if (failure === "unavailable")
            return new HttpResponse(null, { status: 503 });
          return HttpResponse.json({
            data: failure === "empty" ? [] : [{ name: "not-an-id" }],
          });
        }),
      );
      vi.spyOn(ConfigService, "searchModels").mockResolvedValue({
        items: [
          {
            provider: "openrouter",
            name: "vendor/cached-model",
            verified: false,
            free: false,
            default: false,
          },
        ],
        next_page_id: null,
      });
      const { result } = renderHook(() => useProviderModels("openrouter"), {
        wrapper: createWrapper(),
      });
      await waitFor(() =>
        expect(result.current.data?.map((model) => model.name)).toEqual([
          "vendor/cached-model",
        ]),
      );
    },
  );

  it("still falls back to backend discovery when the live catalog AND verification both fail", async () => {
    server.use(
      http.get(
        "https://openrouter.ai/api/v1/models",
        () => new HttpResponse(null, { status: 503 }),
      ),
      http.get(
        "/api/llm/models/verified",
        () => new HttpResponse(null, { status: 503 }),
      ),
    );
    vi.spyOn(ConfigService, "searchModels").mockResolvedValue({
      items: [
        {
          provider: "openrouter",
          name: "vendor/cached-model",
          verified: false,
          free: false,
          default: false,
        },
      ],
      next_page_id: null,
    });
    const { result } = renderHook(() => useProviderModels("openrouter"), {
      wrapper: createWrapper(),
    });
    await waitFor(() =>
      expect(result.current.data?.map((model) => model.name)).toEqual([
        "vendor/cached-model",
      ]),
    );
    expect(result.current.error).toBeFalsy();
  });

  it("discovers live models even when backend verification metadata is unavailable", async () => {
    server.use(
      http.get("https://openrouter.ai/api/v1/models", () =>
        HttpResponse.json({ data: [{ id: "vendor/new-model" }] }),
      ),
      http.get(
        "/api/llm/models/verified",
        () => new HttpResponse(null, { status: 503 }),
      ),
    );
    const { result } = renderHook(() => useProviderModels("openrouter"), {
      wrapper: createWrapper(),
    });
    await waitFor(() =>
      expect(result.current.data).toEqual([
        {
          provider: "openrouter",
          name: "vendor/new-model",
          verified: false,
          free: false,
          default: false,
        },
      ]),
    );
  });

  it("does not block the public catalog on a slow (or hanging) verification lookup, and merges verified flags in once it resolves", async () => {
    let resolveVerified: (value: unknown) => void = () => {};
    const verifiedResponse = new Promise((resolve) => {
      resolveVerified = resolve;
    });
    server.use(
      http.get("https://openrouter.ai/api/v1/models", () =>
        HttpResponse.json({ data: [{ id: "vendor/slow-verify-model" }] }),
      ),
      http.get("/api/llm/models/verified", async () => {
        // Simulate a verification round-trip that never completes within any
        // reasonable UI wait — the catalog must not wait on this.
        await verifiedResponse;
        return HttpResponse.json({
          models: { openrouter: ["vendor/slow-verify-model"] },
        });
      }),
    );

    const { result } = renderHook(() => useProviderModels("openrouter"), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.[0]?.verified).toBe(false);

    resolveVerified(undefined);

    await waitFor(() => expect(result.current.data?.[0]?.verified).toBe(true));
  });

  it("surfaces OpenRouter capability metadata for display without inventing missing fields", async () => {
    server.use(
      http.get("https://openrouter.ai/api/v1/models", () =>
        HttpResponse.json({
          data: [
            {
              id: "vendor/tool-model",
              context_length: 128000,
              supported_parameters: ["tools", "reasoning"],
              architecture: {
                input_modalities: ["text"],
                output_modalities: ["text"],
              },
              top_provider: { max_completion_tokens: 4096 },
              pricing: { prompt: "0.000003", completion: "0.000015" },
            },
            { id: "vendor/bare-model" },
          ],
        }),
      ),
    );
    const { result } = renderHook(() => useProviderModels("openrouter"), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const [withCapabilities, bare] = result.current.data ?? [];
    expect(withCapabilities).toEqual({
      provider: "openrouter",
      name: "vendor/tool-model",
      verified: false,
      free: false,
      default: false,
      capabilities: {
        supportedParameters: ["tools", "reasoning"],
        inputModalities: ["text"],
        outputModalities: ["text"],
        contextLength: 128000,
        maxOutputTokens: 4096,
        pricing: { prompt: "0.000003", completion: "0.000015" },
      },
    });
    // No metadata at all -> `capabilities` is omitted, not `{}`.
    expect(bare).toEqual({
      provider: "openrouter",
      name: "vendor/bare-model",
      verified: false,
      free: false,
      default: false,
    });
    expect(bare.capabilities).toBeUndefined();
  });

  it("leaves other providers on backend discovery", async () => {
    let requests = 0;
    server.use(
      http.get("https://openrouter.ai/api/v1/models", () => {
        requests += 1;
        return HttpResponse.json({ data: [] });
      }),
    );
    const { result } = renderHook(() => useProviderModels("anthropic"), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.length).toBeGreaterThan(0);
    expect(requests).toBe(0);
  });
});
