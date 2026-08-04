import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import LLMBalanceService from "./llm-balance-service";

const localBackend: Backend = {
  id: "local-test",
  name: "Local test backend",
  host: "http://localhost:3000",
  apiKey: "test-session-key",
  kind: "local",
};

const balancePayload = {
  provider: "openrouter",
  limit: 20,
  limit_remaining: 14.96,
  usage: 5.04,
  usage_daily: 1.2,
  usage_weekly: 3.4,
  usage_monthly: 5.04,
  is_free_tier: false,
};

function mockFetchResponse(status: number, body?: unknown) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  });
}

describe("LLMBalanceService.getBalance", () => {
  beforeEach(() => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    setActiveSelection(null);
    setRegisteredBackends([]);
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("normalizes a successful balance response", async () => {
    const fetchMock = mockFetchResponse(200, balancePayload);
    vi.stubGlobal("fetch", fetchMock);

    const balance = await LLMBalanceService.getBalance();

    expect(balance).toEqual({
      provider: "openrouter",
      limit: 20,
      limitRemaining: 14.96,
      usage: 5.04,
      usageDaily: 1.2,
      usageWeekly: 3.4,
      usageMonthly: 5.04,
      isFreeTier: false,
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/llm/balance");
    expect((init.headers as Headers).get("X-Session-API-Key")).toBe(
      "test-session-key",
    );
  });

  it("treats null caps as an uncapped key", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetchResponse(200, {
        ...balancePayload,
        limit: null,
        limit_remaining: null,
      }),
    );

    const balance = await LLMBalanceService.getBalance();

    expect(balance?.limit).toBeNull();
    expect(balance?.limitRemaining).toBeNull();
    expect(balance?.usage).toBe(5.04);
  });

  it("returns null when the endpoint is missing (404)", async () => {
    vi.stubGlobal("fetch", mockFetchResponse(404));

    await expect(LLMBalanceService.getBalance()).resolves.toBeNull();
  });

  it("returns null for a malformed response body", async () => {
    vi.stubGlobal("fetch", mockFetchResponse(200, { unexpected: true }));

    await expect(LLMBalanceService.getBalance()).resolves.toBeNull();
  });

  it("throws on non-404 HTTP failures", async () => {
    vi.stubGlobal("fetch", mockFetchResponse(500));

    await expect(LLMBalanceService.getBalance()).rejects.toThrow(
      "Balance request failed with 500",
    );
  });
});
