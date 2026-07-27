import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import EventService from "#/api/event-service/event-service.api";
import { callCloudProxy } from "#/api/cloud/proxy";
import type { Backend } from "#/api/backend-registry/types";

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-key",
  kind: "cloud",
};

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
  vi.mocked(callCloudProxy).mockReset();
  vi.mocked(callCloudProxy).mockResolvedValue({
    items: [],
    next_page_id: null,
  });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(callCloudProxy).mockReset();
});

describe("EventService.searchEvents — cloud branch", () => {
  it("forwards all pagination params to the cloud proxy and clamps limit to <=100", async () => {
    const options = {
      limit: 500,
      sortOrder: "TIMESTAMP_DESC" as const,
      pageId: "p1",
      timestampGte: "2026-05-01T00:00:00.000000",
      timestampLt: "2026-05-12T07:20:29.087853",
    };

    await EventService.searchEvents("conv-1", null, null, options);

    const proxyCall = vi.mocked(callCloudProxy).mock.calls[0][0];
    const url = new URL(`https://x${proxyCall.path}`);
    expect(url.searchParams.get("limit")).toBe("100");
    expect(url.searchParams.get("sort_order")).toBe("TIMESTAMP_DESC");
    expect(url.searchParams.get("page_id")).toBe("p1");
    expect(url.searchParams.get("timestamp__gte")).toBe(
      "2026-05-01T00:00:00.000000",
    );
    expect(url.searchParams.get("timestamp__lt")).toBe(
      "2026-05-12T07:20:29.087853",
    );
  });

  it("sends only limit when no filter params are provided", async () => {
    await EventService.searchEvents("conv-1", null, null, { limit: 50 });

    const proxyCall = vi.mocked(callCloudProxy).mock.calls[0][0];
    expect(proxyCall.path).toBe(
      "/api/v1/conversation/conv-1/events/search?limit=50",
    );
  });

  it("returns empty page when full-param request fails (graceful degradation)", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    vi.mocked(callCloudProxy).mockRejectedValueOnce(
      new Error("Internal Server Error"),
    );

    const result = await EventService.searchEvents("conv-1", null, null, {
      limit: 50,
      sortOrder: "TIMESTAMP_DESC",
      timestampLt: "2026-05-12T00:00:00.000000",
    });

    // Only one call — no retry, just returns empty page to stop pagination.
    expect(vi.mocked(callCloudProxy)).toHaveBeenCalledTimes(1);
    expect(result.items).toHaveLength(0);
    expect(result.next_page_id).toBeNull();
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("doesn't support pagination filters"),
    );

    warnSpy.mockRestore();
  });

  it("rethrows when a limit-only request (no filter params) fails", async () => {
    vi.mocked(callCloudProxy).mockRejectedValueOnce(
      new Error("Network error"),
    );

    await expect(
      EventService.searchEvents("conv-1", null, null, { limit: 50 }),
    ).rejects.toThrow("Network error");

    expect(vi.mocked(callCloudProxy)).toHaveBeenCalledTimes(1);
  });

  it("stops pagination when server returns fewer items than limit", async () => {
    vi.mocked(callCloudProxy).mockResolvedValueOnce({
      items: [{ id: "evt-1" }, { id: "evt-2" }],
      next_page_id: null,
    });

    const result = await EventService.searchEvents("conv-1", null, null, {
      limit: 50,
      sortOrder: "TIMESTAMP_DESC",
    });

    expect(result.items).toHaveLength(2);
    expect(result.next_page_id).toBeNull();
  });
});
