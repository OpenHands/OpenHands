import { afterEach, describe, expect, it, vi } from "vitest";
import { SSYCLOUD_MODELS_URL } from "#/constants/ssycloud";
import { fetchSSYCloudModels } from "./ssycloud-service.api";

function stubFetchResponse(response: Partial<Response>) {
  const fetchMock = vi.fn().mockResolvedValue(response as Response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchSSYCloudModels", () => {
  it("fetches, filters, deduplicates, and sorts chat-completions models", async () => {
    const fetchMock = stubFetchResponse({
      ok: true,
      json: async () => ({
        data: [
          {
            id: "openai/gpt-5.2",
            support_apis: ["/v1/chat/completions", "/v1/responses"],
          },
          {
            id: "deepseek/deepseek-v4-flash",
            support_apis: ["/v1/chat/completions"],
          },
          {
            id: "openai/gpt-5.2",
            support_apis: ["/v1/chat/completions"],
          },
          { id: "openai/responses-only", support_apis: ["/v1/responses"] },
          // Older responses may omit support_apis; retain those models.
          { id: "anthropic/claude-sonnet-4.6" },
          { id: 42 },
        ],
      }),
    });
    const controller = new AbortController();

    await expect(
      fetchSSYCloudModels("  test-api-key  ", controller.signal),
    ).resolves.toEqual([
      {
        provider: "ssycloud",
        name: "anthropic/claude-sonnet-4.6",
        verified: false,
      },
      {
        provider: "ssycloud",
        name: "deepseek/deepseek-v4-flash",
        verified: false,
      },
      {
        provider: "ssycloud",
        name: "openai/gpt-5.2",
        verified: false,
      },
    ]);

    expect(fetchMock).toHaveBeenCalledWith(SSYCLOUD_MODELS_URL, {
      method: "GET",
      headers: {
        Accept: "application/json",
        Authorization: "Bearer test-api-key",
      },
      signal: controller.signal,
    });
  });

  it("does not make a request without a usable API key", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchSSYCloudModels("   ")).resolves.toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects failed and malformed responses", async () => {
    stubFetchResponse({ ok: false, status: 401 });
    await expect(fetchSSYCloudModels("bad-key")).rejects.toThrow("401");

    vi.unstubAllGlobals();
    stubFetchResponse({ ok: true, json: async () => ({ object: "list" }) });
    await expect(fetchSSYCloudModels("test-key")).rejects.toThrow(
      "missing data",
    );
  });
});
