import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchLatestAgentCanvasVersion } from "./agent-canvas-updates";

function stubFetchResponse(response: Partial<Response>) {
  const fetchMock = vi.fn().mockResolvedValue(response as Response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchLatestAgentCanvasVersion", () => {
  it("fetches the npm registry latest dist-tag and returns the trimmed version", async () => {
    const fetchMock = stubFetchResponse({
      ok: true,
      json: async () => ({ version: " 1.6.0 " }),
    });

    await expect(fetchLatestAgentCanvasVersion()).resolves.toBe("1.6.0");
    expect(fetchMock).toHaveBeenCalledWith(
      "https://registry.npmjs.org/@openhands/agent-canvas/latest",
      expect.objectContaining({ headers: { Accept: "application/json" } }),
    );
  });

  it("rejects when the registry responds with an error status", async () => {
    stubFetchResponse({ ok: false, status: 503 });

    await expect(fetchLatestAgentCanvasVersion()).rejects.toThrow("503");
  });

  it("rejects when the response has no usable version field", async () => {
    stubFetchResponse({ ok: true, json: async () => ({ version: 42 }) });

    await expect(fetchLatestAgentCanvasVersion()).rejects.toThrow(
      "missing version",
    );
  });
});
